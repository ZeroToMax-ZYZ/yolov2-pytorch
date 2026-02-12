from nets.yolov2 import YOLOv2

def build_model(cfg):
    model_name = cfg["model_name"]

    if model_name == "YOLOv2":
        model = YOLOv2(num_classes=20, num_anchors=5, ic_debug=False, reshape_output=True)

    else:
        raise ValueError(f"❗Unsupported model name: {model_name}")
    
    return model