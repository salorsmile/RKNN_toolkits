from rknn.api import RKNN
import argparse


class RKNN_model_container():
    # 初始化模型
    def __init__(self, args) -> None:
        self.perf_debug = args.perf_debug
        self.eval_mem = args.eval_mem
        self.target = args.target
        self.device_id = args.device_id
        self.rknn = RKNN()

        try:
            # Load RKNN Model
            self.rknn.load_rknn(args.model_path)
            print('--> Init runtime environment')

            if self.target is None:
                ret = self.rknn.init_runtime(perf_debug=self.perf_debug, eval_mem=self.eval_mem)
            else:
                ret = self.rknn.init_runtime(target=self.target,
                                             device_id=self.device_id,
                                             perf_debug=self.perf_debug,
                                             eval_mem=self.eval_mem)
            if ret != 0:
                raise RuntimeError("Init runtime environment failed")
            print('RKNN model loaded.')

        except Exception as e:
            print(f"Error during initialization: {e}")
            exit(1)

    # 执行一次图像推理
    def run(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        try:
            result = self.rknn.inference(inputs=inputs)
            return result
        except Exception as e:
            print(f"Error during inference: {e}")
            return None

    def release(self):
        if self.rknn:
            self.rknn.release()

    def print_model_perf(self):
        try:
            perf_detail = self.rknn.eval_perf()
            mem_detail = self.rknn.eval_memory()
            print('RKNN model runtime performance:', perf_detail, mem_detail)
        except Exception as e:
            print(f"Error during performance evaluation: {e}")

    def print_sdk_version(self):
        print('RKNN runtime sdk version:', self.rknn.get_sdk_version())
