# ------------------------------------------------------------------------
# Conditional DETR for Image Cropping
# ------------------------------------------------------------------------
# Modified from ConditionalDETR (https://github.com/Atten4Vis/ConditionalDETR)
# ------------------------------------------------------------------------

import math
import torch
import torch.nn.functional as F
from torch import nn
import copy
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
from sklearn.metrics import roc_auc_score, roc_curve
from .position_encoding import build_position_encoding
from typing import Any, Optional, Tuple, Type
from .attention_embedding import Attention
from .networks_ema import update_moving_average, EMA, singleton

class ConditionalDETR(nn.Module):
    """ This is the Conditional DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, args=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.args = args


        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
    
        
    def gen_sineembed_for_position(self, pos_tensor):
        # n_query, bs, _ = pos_tensor.size()
        # sineembed_tensor = torch.zeros(n_query, bs, 256)
        scale = 2 * math.pi
        dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / 128)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        return pos

    def forward(self, samples, samples_retrieves, boxes_retrieve, sam_embedding, mask):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
            if mask is not None:
                samples.mask = mask
        
        # srcs_retrieves = []
        # masks_retrieves = []
        # poss_retrieves = []
        
        # sample_retrieve = nested_tensor_from_tensor_list(samples_retrieves.flatten(0,1))
        # with torch.no_grad():
        #     features_retrieve, poss_retrieves = self.backbone(sample_retrieve)
        #     srcs_retrieves, masks_retrieves = features_retrieve[-1].decompose()
        #     srcs_retrieves = self.input_proj(srcs_retrieves)

        # crop_embedding = self.gen_sineembed_for_position(boxes_retrieve)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        # print("input shape:", src.shape)
        src = self.input_proj(src)
        # print("output shape:", src.shape, mask.shape, pos[-1].shape)
        assert mask is not None
        
        # hs, reference = self.transformer(src, srcs_retrieves,  mask, masks_retrieves, self.query_embed.weight, pos[-1], poss_retrieves[-1], crop_embedding, sam_embedding)
        hs, reference = self.transformer(src, None,  mask, None, self.query_embed.weight, pos[-1], None, samples_retrieves, sam_embedding)
        
        # hs, reference = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])
        
        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            tmp = self.bbox_embed(hs[lvl])
            tmp[..., :2] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
            
        outputs_coord = torch.stack(outputs_coords)
        outputs_class = self.class_embed(hs)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

def post_process_gt(batchgt, scale=False):
    processed_batchgt = []
    for datadict in batchgt:
        # datadict['scores'] = datadict['scores'] * self.score_std + self.score_mean
        # datadict['good_scores'] = datadict['good_scores'] * self.score_std + self.score_mean
        orig_size = datadict['orig_size']
        w, h = orig_size[0], orig_size[1]
        norm_factor = torch.tensor([w, h, w, h], dtype=torch.float32).to(datadict['boxes'].device)
        # minscale = torch.as_tensor([0.35, 0.35, 0.55, 0.55]).to(datadict['boxes'].device)
        # maxscale = torch.as_tensor([0.65, 0.65, 0.95, 0.95]).to(datadict['boxes'].device)
        minscale = torch.as_tensor([0., 0., 0., 0.]).to(datadict['boxes'].device)
        maxscale = torch.as_tensor([1., 1., 1., 1.]).to(datadict['boxes'].device)
        datadict['boxes'] = datadict['boxes'] * (maxscale - minscale) + minscale
        datadict['good_boxes'] = datadict['good_boxes'] * (maxscale - minscale) + minscale
        datadict['boxes'] = box_ops.box_cxcywh_to_xyxy(datadict['boxes'])
        datadict['good_boxes'] = box_ops.box_cxcywh_to_xyxy(datadict['good_boxes'])
        if scale:
            datadict['boxes'] = datadict['boxes'] * norm_factor
            datadict['boxes']= torch.clamp(datadict['boxes'], min=torch.as_tensor(0).cuda())
            datadict['boxes'][:,2] = torch.clamp(datadict['boxes'][:,2], max=torch.as_tensor(w).cuda())
            datadict['boxes'][:,3] = torch.clamp(datadict['boxes'][:,3], max=torch.as_tensor(h).cuda())
            datadict['good_boxes'] = datadict['good_boxes'] * norm_factor
            datadict['good_boxes']= torch.clamp(datadict['good_boxes'], min=torch.as_tensor(0).cuda())
            datadict['good_boxes'][:,2] = torch.clamp(datadict['good_boxes'][:,2], max=torch.as_tensor(w).cuda())
            datadict['good_boxes'][:,3] = torch.clamp(datadict['good_boxes'][:,3], max=torch.as_tensor(h).cuda())
        processed_batchgt.append(datadict)
    return processed_batchgt

class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, args):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = args.focal_alpha
        self.focal_gamma = args.focal_gamma
        self.auclist = []
        self.acclist = []
        self.soft_iou_thresh = args.soft_iou_thresh
        self.soft_bound = torch.tensor([args.soft_bound])
        self.use_valid_smooth = args.use_valid_smooth
        self.indices = None
        
    def soft_label(self, outputs, targets):

        boxes = box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes'])

        targets_copy = copy.deepcopy(targets)
        targets_processed = post_process_gt(targets_copy, scale=False)

        iou_all = []
        soft_label_all = []
        for idx, gt_dict in enumerate(targets_processed):
            gt_boxes = gt_dict['boxes']
            pr_boxes = boxes[idx].type_as(gt_boxes)

            iou_mat = box_ops.box_iou(pr_boxes, gt_boxes)[0]
            iou_all.append(iou_mat)

            maxiou_for_each_pr, maxiou_indices = torch.max(iou_mat, dim=1)
            valid_indices = (maxiou_for_each_pr >= self.soft_iou_thresh).type_as(maxiou_for_each_pr)
            quality_pr = gt_dict['scores'][maxiou_indices] * valid_indices

            soft_label = torch.clamp(self.soft_bound.item() * (quality_pr - 2) / 1.5, min=0, max=self.soft_bound.item())

            soft_label_all.append(soft_label)

        soft_label_all = torch.stack(soft_label_all, 0)

        return soft_label_all

    def soft_label_valid(self, outputs, targets, indices):

        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat([t['good_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        cropscale = target_boxes[:, 2] * target_boxes[:, 3]

        label_base = torch.full(src_logits.shape[:2], 1, dtype=cropscale.dtype, device=src_logits.device)
        label_valid = (cropscale > 0.75) * 0.9 + (1 - (cropscale > 0.75).type_as(label_base))

        label_base[idx] = label_valid

        label_base = label_base.unsqueeze(2)

        return label_base

    def loss_labels(self, outputs, targets, indices, num_boxes, epoch, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        if self.soft_bound > 0:
            soft_label_full = self.soft_label(outputs, targets).unsqueeze(2)
            target_classes_onehot = (1 - target_classes_onehot) * soft_label_full + target_classes_onehot

        if self.use_valid_smooth:
            label_base = self.soft_label_valid(outputs, targets, indices)
            target_classes_onehot = target_classes_onehot * label_base
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                     alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, epoch):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, epoch):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['good_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses


    def loss_masks(self, outputs, targets, indices, num_boxes, epoch):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, epoch, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, epoch, **kwargs)

    def forward(self, outputs, targets, epoch):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        
        try:
            indices = self.matcher(outputs_without_aux, targets)
            self.indices = indices
        except:
            indices = self.indices
        
        # indices = self.matcher(outputs_without_aux, targets)
        
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["good_scores"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, epoch))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, epoch, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, topk_num=100):
        super().__init__()
        self.topk_num = topk_num

    def forward(self, outputs, target_sizes, use_aux=-1):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # if use_aux < 0 :
        #     out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        # else:
        #     out_logits, out_bbox = outputs['aux_outputs'][use_aux]['pred_logits'], \
        #                            outputs['aux_outputs'][use_aux]['pred_boxes']
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        # assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        scores_oriorder = prob.view(out_logits.shape[0], -1)

        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.topk_num, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes_oriorder = out_bbox
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        # img_h, img_w = target_sizes.unbind(1)
        img_w, img_h = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).cuda()
        boxes = boxes * scale_fct[:, None, :]
        # boxes = torch.clamp(boxes, min=torch.as_tensor(0).cuda())
        # for i in range(boxes.shape[0]):
        #     print(boxes[i,:,2], img_w[i])
        #     boxes[i,:,2] = torch.clamp(boxes[i,:,2], max=torch.as_tensor(img_w[i]).cuda())
        #     boxes[i,:,3] = torch.clamp(boxes[i,:,3], max=torch.as_tensor(img_h[i]).cuda())


        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        # results_oriorder = [{'scores': s, 'boxes': b} for s, b in zip(scores_oriorder, boxes_oriorder)]

        return results, (topk_values, topk_indexes)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.norms =  nn.ModuleList(nn.LayerNorm(k) for _, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if i < self.num_layers - 1:
                x = self.norms[i](x)
        return x

class ConditionalDETR_distill(nn.Module):
    """ This is the Conditional DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, args):
        super().__init__()

        self.conditionalDETR = ConditionalDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,)

        self.netT = None
        self.ema_updater = EMA(args.moving_average_decay)

    def forward(self, samples, samples_retrieves, boxes_retrieve, sam_embedding, mask):
        return self.conditionalDETR(samples, samples_retrieves, boxes_retrieve, sam_embedding, mask)

    @singleton('netT')
    def _get_target_network(self):
        netT = copy.deepcopy(self.conditionalDETR)
        return netT

    def reset_moving_average(self):
        del self.netT
        self.netT = None

    def update_moving_average(self):
        assert self.netT is not None, 'target network has not been created yet'
        update_moving_average(self.ema_updater, self.netT, self.conditionalDETR)

    def produce_pseudo_label(self, samples, samples_retrieves, boxes_retrieve, sam_embedding):
        with torch.no_grad():
            self.netT = self._get_target_network()
            return self.netT(samples, samples_retrieves, boxes_retrieve, sam_embedding)



# def build(args):
#     # the `num_classes` naming here is somewhat misleading.
#     # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
#     # is the maximum id for a class in your dataset. For example,
#     # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
#     # As another example, for a dataset that has a single class with id 1,
#     # you should pass `num_classes` to be 2 (max_obj_id + 1).
#     # For more details on this, check the following discussion
#     # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
#     num_classes = 1
#     device = torch.device(args.device)

#     backbone = build_backbone(args)

#     transformer = build_transformer(args)

#     model = ConditionalDETR(
#         backbone,
#         transformer,
#         num_classes=num_classes,
#         num_queries=args.num_queries,
#         aux_loss=args.aux_loss,
#         args = args
#     )

#     matcher = build_matcher(args)
#     weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
#     weight_dict['loss_giou'] = args.giou_loss_coef

#     # TODO this is a hack
#     if args.aux_loss:
#         aux_weight_dict = {}
#         for i in range(args.dec_layers - 1):
#             aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
#         weight_dict.update(aux_weight_dict)


#     losses = ['labels', 'boxes']

#     criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses, args=args)
#     criterion.cuda()
#     postprocessors = {'bbox': PostProcess(topk_num=args.topk_num)}


#     return model, criterion, postprocessors



def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 1
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = ConditionalDETR_distill(
        backbone,
        transformer,
        num_classes=num_classes,
        args=args
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses, args=args)
    criterion.cuda()
    postprocessors = {'bbox': PostProcess(topk_num=args.topk_num)}
    postprocessors_flms = {'bbox': PostProcess(topk_num=args.topk_num)}

    return model, criterion, postprocessors, postprocessors_flms
