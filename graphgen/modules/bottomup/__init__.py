"""Package containing the Bottom-Up VQA model by Anderson et. al.

References:
-----------
https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Bottom-Up_and_\
Top-Down_CVPR_2018_paper.html
"""

from .network import BottomUp as BottomUp

__all__ = [BottomUp.__name__]
