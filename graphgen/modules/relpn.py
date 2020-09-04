"""Implementation of a relation proposal network."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision


class SpatialRelPN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B905
    """Implementation of a relation proposal network."""

    def __init__(self, feature_dim: int):
        """Construct a `RelPN` instance."""
        super().__init__()
        self.similarity = RelPNSimilarity(feature_dim)

    def _relatedness(
        self,
        feats: List[torch.Tensor],
        proposals: List[torch.Tensor],
        take: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute a relatedness score for every pair of boxes based on their \
        class logits.

        Params:
        -------
        `class_logits`: pre-softmax outputs of the box head classifier for all
        boxes for all images.
        `proposals`: List of tensors of shape `(num_boxes, 4)`, the bounding \
        box proposals in the original image space, for each image.

        Returns:
        --------
        `all_pairs`: A list of lists of pairs of indices. The `i`th element of
        the outer list refers to the list of index pairs for the `i`th image.
        Each index pair points to the subject and object bounding box indices in
        the `proposals` tensor. Index pairs are aligned with their corresponding
        relatedness score, stored in `all_scores`.
        `all_scores`: A list of lists of relatedness scores for each (subject, object)
        proposal pair, in descending order for each image.
        """
        all_pairs = []
        all_scores = []
        for img_logits, img_proposals in zip(feats, proposals):
            # Get relatedness scores between class logits. Entry i, j refers to
            # relatedness between subject i and object j
            relatedness = self.similarity(img_logits)

            # Create a matrix of i, j indices that we can look up later
            meshgrid = torch.transpose(
                torch.tensor(  # pylint: disable=not-callable
                    np.meshgrid(
                        np.arange(relatedness.size(0)),
                        np.arange(relatedness.size(1)),
                        indexing="xy",
                    )
                ),
                0,
                -1,
            )  # TODO consider if we can keep a meshgrid with no grad and share it?
            meshgrid = meshgrid.reshape(-1, 2).to("cuda")

            # Assert meshgrid matches up with relatedness after changing the view.
            # for idx in range(relatedness.view(-1).size(0)):
            #     assert (
            #         relatedness.view(-1)[idx]
            #         == relatedness[meshgrid[idx][0]][meshgrid[idx][1]]
            #     )

            # Take the top `take` pairs.
            scores, indices = torch.sort(relatedness.view(-1), descending=True)
            if take is not None:
                scores = scores[:take]
                indices = indices[:take]

            # Map sorted indices back to original indices
            index_pairs = torch.index_select(meshgrid, dim=0, index=indices)

            # Assert looking up original relatedness score with retrieved index
            # pairs yields the correct score.
            # for score, idx_pair in zip(scores, index_pairs):
            #     assert score == relatedness[idx_pair[0]][idx_pair[1]]

            all_pairs.append(index_pairs)
            all_scores.append(scores)

        return all_pairs, all_scores

    # def filter_overlaps(  # pylint: disable=no-self-use
    #     self,
    #     proposals: List[torch.Tensor],
    #     relation_pairs: List[torch.Tensor],
    #     relation_scores: List[torch.Tensor],
    # ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    #     """Use non-maximal supression to filter pairs of bounding boxes whose \
    #     subject and object bounding boxes are too similar."""
    #     # pylint: disable=too-many-locals
    #     nms_relation_pairs = []
    #     nms_relation_scores = []
    #     for img_proposals, img_relation_pairs, img_relation_scores in zip(
    #         proposals, relation_pairs, relation_scores
    #     ):
    #         subjects = img_proposals[img_relation_pairs[:, 0], :]
    #         objects = img_proposals[img_relation_pairs[:, 1], :]

    #         # for idx, sub in zip(img_relation_pairs[:, 0], subjects):
    #         #     assert torch.equal(sub, img_proposals[idx])
    #         # for idx, obj in zip(img_relation_pairs[:, 1], objects):
    #         #     assert torch.equal(obj, img_proposals[idx])

    #         # Calculate IoU for each box in proposals with each box in targets
    #         subj_ious = torchvision.ops.boxes.box_iou(
    #             subjects, subjects
    #         )  # ious[i][j] is iou between subject i and object j
    #         obj_ious = torchvision.ops.boxes.box_iou(
    #             objects, objects
    #         )  # ious[i][j] is iou between subject i and object j

    #         # TODO more efficient NMS
    #         suppress_indices = []
    #         iou_threshold = 0.5
    #         for i in range(img_relation_pairs.size(0)):
    #             for j in range(i + 1, img_relation_pairs.size(0)):
    #                 pair_iou = (
    #                     subj_ious[i][j] * obj_ious[i][j]
    #                 )  # Different to grcnn paper, we multiply instead of add
    #                 if pair_iou > iou_threshold:
    #                     suppress_indices.append(
    #                         i if img_relation_scores[i] < img_relation_scores[j]
    #                         else j
    #                     )
    #         mask = torch.tensor(  # pylint: disable=not-callable
    #             [i in suppress_indices for i in range(img_relation_pairs.size(0))]
    #         )
    #         nms_relation_pairs.append(img_relation_pairs[mask, :])
    #         nms_relation_scores.append(img_relation_scores[mask])

    #     return nms_relation_pairs, nms_relation_scores

    def box_intersect(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Return intersection of boxes.

        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

        Arguments:
        ----------
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

        Returns:
        --------
        intersections (Tensor[N, M, 4]): the NxMx4 matrix containing the
        intersections of each box in boxes1 with each box in boxes2.

        References:
        -----------
        https://pytorch.org/docs/stable/_modules/torchvision/ops/boxes.html#nms
        """
        x1_y1 = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2] left-top
        x2_y2 = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2] right-bottom
        return torch.cat((x1_y1, x2_y2), dim=2)

    def forward(
        self,
        feats: List[torch.Tensor],
        proposals: List[torch.Tensor],
        pre_nms_limit: Optional[int] = 256,
        iou_threshold: float = 0.7,
        post_nms_limit: Optional[int] = 128,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Return a set of relation pairs and scores for each image.

        Params:
        -------
        `feats`: List of tensors of shape `(num_proposals, feature_dim)`.
        `proposals`: List of tensors each of shape `(num_boxes, 4)`,
        the bounding box proposals in the original image space for each image.
        """
        relation_pairs, relation_scores = self._relatedness(
            feats, proposals, take=pre_nms_limit
        )
        # Get intersections between box pairs, representing the spatial interaction
        # of two objects
        intersections = [
            self.box_intersect(img_proposals, img_proposals)
            for img_proposals in proposals
        ]  # TODO experiment with more than intersections
        # print(f"{[img_intersections.size() for img_intersections in intersections]=}")

        # Take only the intersections indexed by `relation_pairs`
        pre_nms_intersections = [
            torch.stack([img_intersections[i][j] for i, j in img_relation_pairs], dim=0)
            for img_relation_pairs, img_intersections in zip(
                relation_pairs, intersections
            )
        ]  # TODO make more efficient, maybe with torch.gather

        # Perform NMS on object intsersections
        keep = [
            torchvision.ops.boxes.nms(
                img_pre_nms_intersections, img_relation_scores, iou_threshold
            )[:post_nms_limit]
            for img_pre_nms_intersections, img_relation_scores in zip(
                pre_nms_intersections, relation_scores
            )
        ]  # TODO migrate to batched_nms instead of nms for each image
        post_nms_intersections = [
            torch.index_select(img_pre_nms_intersections, dim=0, index=img_keep)
            for img_keep, img_pre_nms_intersections in zip(keep, pre_nms_intersections)
        ]
        post_nms_relation_pairs = [
            torch.index_select(img_relation_pairs, dim=0, index=img_keep)
            for img_keep, img_relation_pairs in zip(keep, relation_pairs)
        ]
        post_nms_relation_scores = [
            torch.index_select(img_relation_scores, dim=0, index=img_keep)
            for img_keep, img_relation_scores in zip(keep, relation_scores)
        ]

        return post_nms_relation_pairs, post_nms_relation_scores, post_nms_intersections


class RelPNSimilarity(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B905
    """Relation Proposal Network Similarity from "Graph R-CNN for Scene Graph \
    Generation".

    References:
    -----------
    Graph R-CNN for Scene Graph Generation.
    Jianwei Yang, Jiasen Lu, Stefan Lee, Dhruv Batra and Devi Parikh.
    https://arxiv.org/pdf/1808.00191.pdf
    """

    def __init__(self, in_features: int) -> None:
        """Create a RelSimilarity module."""
        super().__init__()
        # TODO positional encoding?
        # https://github.com/alexmirrington/graph-rcnn.pytorch/blob/master/
        # lib/scene_parser/rcnn/modeling/relation_heads/relpn/relpn.py
        self.subject_projection = torch.nn.Sequential(  # phi in the paper
            torch.nn.Linear(in_features, 256),  # 64 in the paper
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 64),
        )
        self.object_projection = torch.nn.Sequential(  # psi in the paper
            torch.nn.Linear(in_features, 256),  # 64 in the paper
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 64),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Propagate data through the model."""
        subj_data = self.subject_projection(data)  # k x 64, k is number of feats
        obj_data = self.object_projection(data)  # k x 64
        scores = torch.mm(subj_data, obj_data.t())  # k x k
        return torch.sigmoid(scores)  # Sigmoid for 0-1 clamp
