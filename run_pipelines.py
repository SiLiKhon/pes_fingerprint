import sys, os, json

SE_PP_PATH = os.path.abspath(os.path.expanduser("../SE_pp_refactor"))

if SE_PP_PATH not in sys.path:
    sys.path.append(SE_PP_PATH)

from pes_fingerprint import pipelines as ppl

if __name__ == "__main__":
    import tensorflow as tf
    tf.config.experimental.enable_tensor_float_32_execution(False)

    from utils import tf as tfu
    tfu.select_best_gpu()
    tfu.set_memory_growth()

    ids = dict(
        lgps="Li10GeP2S12",
        lips="Li7P3S11_v1",
    )
    cps = dict(
        lgps=[
            "pretrained",
            "../SE_pp_refactor_runs/Li10GeP2S12_testrun/training/00507-0.043672-0.009238-0.034435",
            "../SE_pp_refactor_runs/Li10GeP2S12_testrun/training/00533-0.043249-0.009566-0.033683",
            "../SE_pp_refactor_runs/Li10GeP2S12_testrun/training/00560-0.042665-0.009364-0.033300",
            "../SE_pp_refactor_runs/Li10GeP2S12_testrun/training/00574-0.042474-0.009552-0.032922",
            "../SE_pp_refactor_runs/Li10GeP2S12_testrun/training/00598-0.042147-0.009761-0.032386",
            "../SE_pp_refactor_runs/Li10GeP2S12_testrun/training/00634-0.042113-0.010561-0.031552",
            "../SE_pp_refactor_runs/Li10GeP2S12_testrun/training/00636-0.040893-0.008397-0.032496",
            "../SE_pp_refactor_runs/Li10GeP2S12_testrun/training/00657-0.040328-0.009182-0.031146",
            "../SE_pp_refactor_runs/Li10GeP2S12_testrun/training/00731-0.038292-0.008039-0.030253",
            "../SE_pp_refactor_runs/Li10GeP2S12_testrun/training/00794-0.037069-0.007834-0.029234",
            "../SE_pp_refactor_runs/Li10GeP2S12_testrun/training/00895-0.036784-0.007878-0.028906",
        ],
        lips=[
            "pretrained",
            "../SE_pp_refactor_runs/Li7P3S11_v1_testrun2/training/00518-0.048248-0.008175-0.040072",
            "../SE_pp_refactor_runs/Li7P3S11_v1_testrun2/training/00534-0.047861-0.009280-0.038581",
            "../SE_pp_refactor_runs/Li7P3S11_v1_testrun2/training/00563-0.047116-0.009167-0.037949",
            "../SE_pp_refactor_runs/Li7P3S11_v1_testrun2/training/00618-0.046189-0.009483-0.036706",
            "../SE_pp_refactor_runs/Li7P3S11_v1_testrun2/training/00676-0.044146-0.008571-0.035575",
            "../SE_pp_refactor_runs/Li7P3S11_v1_testrun2/training/00695-0.044083-0.009160-0.034923",
            "../SE_pp_refactor_runs/Li7P3S11_v1_testrun2/training/00743-0.043768-0.009178-0.034589",
            "../SE_pp_refactor_runs/Li7P3S11_v1_testrun2/training/00746-0.042939-0.008932-0.034007",
            "../SE_pp_refactor_runs/Li7P3S11_v1_testrun2/training/00779-0.042689-0.008758-0.033931",
            "../SE_pp_refactor_runs/Li7P3S11_v1_testrun2/training/00783-0.042080-0.008447-0.033633",
            "../SE_pp_refactor_runs/Li7P3S11_v1_testrun2/training/00812-0.041053-0.008224-0.032829",
            "../SE_pp_refactor_runs/Li7P3S11_v1_testrun2/training/00876-0.040982-0.007854-0.033127",
            "../SE_pp_refactor_runs/Li7P3S11_v1_testrun2/training/00898-0.040055-0.008349-0.031706",
            "../SE_pp_refactor_runs/Li7P3S11_v1_testrun2/training/00903-0.038660-0.006905-0.031755",
        ],
    )

    grid_sizes = [
        (20, 20, 20),
        (40, 40, 40),
        (60, 60, 60),
    ]

    configs = []
    for gs in grid_sizes:
        for stru in ["lgps", "lips"]:
            for cp in cps[stru]:
                configs.append(
                    ppl.PESBarrierPipelineConfig(
                        model_checkpoint=cp,
                        dataset_id=ids[stru],
                        grid_size=gs,
                    )
                )

    for config in configs:
        print("\n\n\n")
        print("==============" * 10)
        print("==============" * 10)
        print("==============" * 10)
        print("==============" * 10)
        print("\n\n")
        print("Running pipeline for config:")
        print(json.dumps(config.__dict__, indent=2))
        print("\n")
        ppl.pes_barrier_pipeline(config)
