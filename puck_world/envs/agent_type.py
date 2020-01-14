from enum import Enum, unique


@unique
class AgentType(Enum):
    """Agent 类型枚举类
    """
    # 追逐者
    Chaser = 1
    # 逃跑者
    Runner = 2
