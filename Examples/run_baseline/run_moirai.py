from EasyTSAD.Controller import TSADController

if __name__ == "__main__":
    
    # Create a global controller
    gctrl = TSADController()
    
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["TODS", "UCR", "AIOPS", "NAB", "Yahoo", "WSD"]
    datasets = ["NEK"]
    
    # set datasets path, dirname is the absolute/relative path of dataset.
    
    # Use all curves in datasets:
    gctrl.set_dataset(
        dataset_type="UTS",
        dirname="./datasets",
        datasets=datasets,
    )
    
    # Or specify certain curves in one dataset, 
    # e.g. AIOPS 0efb375b-b902-3661-ab23-9a0bb799f4e3 and ab216663-dcc2-3a24-b1ee-2c3e550e06c9
    # gctrl.set_dataset(
    #     dataset_type="UTS",
    #     dirname="/path/to/datasets",
    #     datasets="AIOPS",
    #     specify_curves=True,
    #     curve_names=[
    #         "0efb375b-b902-3661-ab23-9a0bb799f4e3",
    #         "ab216663-dcc2-3a24-b1ee-2c3e550e06c9"
    #     ]
    # )
    
    
    """============= [EXPERIMENTAL SETTINGS] ============="""
    # Specifying methods and training schemas
    from EasyTSAD.Methods import Moirai
    
    methods = ["Moirai"]
    training_schema = "zero_shot"
    
    for method in methods:
        # run models
        gctrl.run_exps(
            method=method,
            training_schema=training_schema
        )
       
        
    """============= [EVALUATION SETTINGS] ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    # Specifying evaluation protocols
    gctrl.set_evals(
        [
            PointF1PA(),
            EventF1PA(),
            EventF1PA(mode="squeeze")
        ]
    )

    for method in methods:
        gctrl.do_evals(
            method=method,
            training_schema=training_schema
        )
        
        
    """============= [PLOTTING SETTINGS] ============="""
    
    # plot anomaly scores for each curve
    for method in methods:
        gctrl.plots(
            method=method,
            training_schema=training_schema
        )
