from typing import Type
import numpy as np
from EasyTSAD.Evaluations import MetricInterface
from .. import MetricInterface, EvalInterface
from ..Metrics import F1class
import sklearn.metrics

class PointF1(EvalInterface):
    """
    Using Traditional F1 score to evaluate the models.
    """
    def __init__(self) -> None:
        super().__init__()
        self.name = "point-wise f1"
        
    def calc(self, scores, labels, margins) -> type[MetricInterface]:
        '''
        Returns:
         A F1class (Evaluations.Metrics.F1class), including:\n
            best_f1: the value of best f1 score;\n
            precision: corresponding precision value;\n
            recall: corresponding recall value;
        '''
        prec, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true=labels,
                                                                     y_score=scores)
        # Sometimes precision_recall_curve picks a few thresholds that are too high for the data
        # so you end up with points where both precision and recall are zero
        # f1_all = (2 * prec * recall) / (prec + recall)
        # compute maximum f1 score using precision_recall_curve https://stackoverflow.com/a/66549018/974526
        numerator = 2 * recall * prec
        denom = recall + prec   # denominator
        f1_all = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_idx = np.argmax(f1_all)
        
        return F1class(
            name=self.name,
            p=prec[max_idx],
            r=recall[max_idx],
            f1=f1_all[max_idx],
            thres=float(thresholds[max_idx])
        )