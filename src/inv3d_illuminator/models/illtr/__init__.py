from .. import model_factory
from .illtr import LitIllTr
from .illtr_template import LitIllTrTemplate

model_factory.register_model("illtr", LitIllTr)
model_factory.register_model("illtr_template", LitIllTrTemplate)
