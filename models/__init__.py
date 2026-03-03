import importlib
from models.base_model import BaseModel

def find_model_using_name(model_name):
    # 修改核心：增加路径判断逻辑
    # 如果名字里带 'our' 或者 名字是 'CIF_MMIN'，就去 models.our 下面找
    if 'our' in model_name or model_name == 'CIF_MMIN':
        model_filename = "models.our." + model_name + "_model"
    # 如果名字里带 'MISA' (保留原有逻辑，如果有的话)
    elif 'MISA' in model_name:
        model_filename = "models." + model_name + "_model"
    # 默认情况：在 models 根目录下找
    else:
        model_filename = "models." + model_name + "_model"

    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options

def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % (type(instance).__name__))
    return instance