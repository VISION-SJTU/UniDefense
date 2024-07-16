from .abstract_engine import AbstractEngine
from .forgery_engine import ForgeryEngine
from .ocim_engine import OCIMEngine
from .uniattack_engine import UniAttackEngine

ENGINE = {
    'FE': ForgeryEngine,
    'OCIM': OCIMEngine,
    'UE': UniAttackEngine
}

def get_engine(name='UE'):
    print(f"Using engine: '{name}'")
    return ENGINE[name]
