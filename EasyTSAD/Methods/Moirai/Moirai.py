from typing import Dict
import torchinfo
import tqdm
from ...DataFactory import TSData
from ...Exptools import EarlyStoppingTorch
from .. import BaseMethod
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from gluonts.dataset.split import split
from ...DataFactory.TorchDataSet import PredictWindow
from gluonts.dataset.pandas import PandasDataset
import pandas as pd

# FIZME 将下述等参数放于`config.toml`配置文件中
SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
PDT = 1  # prediction length: any positive integer
CTX = 16  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 128  # batch size: any positive integer
# TEST = len(df) - CTX # 使CTX之后的都作为测试，rolling测试，PDT为1, 所以有1073-16=1057个值

class Moirai(BaseMethod):
    def __init__(self, params:dict) -> None:
        super().__init__()
        self.__anomaly_score = None
        
        self.cuda = True
        if self.cuda == True and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("=== Using CUDA ===")
        else:
            if self.cuda == True and not torch.cuda.is_available():
                print("=== CUDA is unavailable ===")
            self.device = torch.device("cpu")
            print("=== Using CPU ===")
            
        # self.p = params["p"]
        # self.batch_size = params["batch_size"]
        # self.p = CTX
        # self.batch_size = BSZ
        
        # Prepare pre-trained model by downloading model weights from huggingface hub
        self.model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}",
                                                local_files_only=True),
            prediction_length=PDT,
            context_length=CTX,
            patch_size=PSZ,
            num_samples=100,    # 最后的平均值从此采样中计算
            target_dim=1,
            feat_dynamic_real_dim=0,    # 此两个参数没使用填0，具体意义TODO
            past_feat_dynamic_real_dim=0,
        )
        self.predictor = self.model.create_predictor(batch_size=BSZ, device=self.device)
    
    def train_valid_phase(self, tsTrain: TSData):
        pass
            
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
       # 不微调
       pass
        
    def test_phase(self, tsData: TSData):
        # test_timestamp 一列没在EasyTSAD原仓库预测过程中使用，因其默认填充好残缺timestamp的值，即有固定的interval
        # 因Moirai的需要，修改TSData类，把test_timestamp加上去。TODO: 好像Moirai也可以没有timestamp作为index列，待确认
        df = pd.DataFrame(index=tsData.test_timestamp, data=tsData.test)
        df.index = pd.to_datetime(df.index, unit='s')   # 间隔刚好3600s，转成datatime，否则integer值不能next(input_it)
        TEST = len(df) - CTX # 使CTX之后的都作为测试，rolling测试，PDT为1, 所以有1073-16=1057个值

        # Convert into GluonTS dataset
        ds = PandasDataset(dict(df), freq='H')

        # Split into train/test set
        train, test_template = split(
            ds, offset=-TEST
        )  # assign last TEST time steps as test set

        # Construct rolling window evaluation
        test_data = test_template.generate_instances(
            prediction_length=PDT,  # number of time steps for each prediction
            windows=TEST // PDT,  # number of windows in rolling window evaluation
            distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
            max_history=CTX
        )

        forecasts = self.predictor.predict(test_data.input)
        forecasts_list = list(tqdm.tqdm(forecasts)) # 这个需要时间，因为预测了所有
        # FIXME，因只有一个值，所以直接用`[0]`将array转为数值
        forecast_mean = np.array([forecast.quantile(0.5)[0] for forecast in forecasts_list])
        label_list = np.array([label['target'][0] for label in test_data.label])    # gluonts这里的label是EasyTSAD的target
        
        scores = (label_list - forecast_mean)**2    # MSE，异常分数值为预测值与GroundTruth的差异量
        assert scores.ndim == 1
        self.__anomaly_score = scores
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def param_statistic(self, save_file):
        pass
        # model_stats = torchinfo.summary(self.model, (self.batch_size, self.p), verbose=0)
        # with open(save_file, 'w') as f:
        #     f.write(str(model_stats))