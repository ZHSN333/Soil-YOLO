"""Soil-YOLO 运行时超参：损失 / 学习率 / 优化器 集中管理。

设计原则:
  1. 禁止 optimizer=auto：会忽略手动 lr0（日志里曾出现 ignoring lr0=0.008）。
  2. 预训练 + 自定义模块：AdamW + 较小 lr0，配合余弦退火更稳。
  3. QFL(β=2) 已聚焦难例，cls 增益不宜再过大，避免分类项淹没定位项。
  4. cls_pw 管类别不均衡；QFL 管难易样本；二者分工，不叠加强 Focal(α)。
"""

from __future__ import annotations


QFL_BETA = 2.0
QFL_BCE_MIX = 0.0
FL_GAMMA = QFL_BETA  
FL_ALPHA = QFL_BCE_MIX  




LOSS_BOX = 7.5
LOSS_CLS = 1.5
LOSS_DFL = 1.5
CLS_PW = 0.5  



OPTIMIZER = "AdamW"

LR0 = 0.001

LRF = 0.01
COS_LR = True
MOMENTUM = 0.9  
WEIGHT_DECAY = 0.0005

WARMUP_EPOCHS = 5.0
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.1


VAL_CONF = 0.10
VAL_IOU = 0.50


def describe_optim_and_loss() -> str:
    """日志摘要。"""
    return (
        f"opt={OPTIMIZER} lr0={LR0} lrf={LRF} cos_lr={COS_LR} "
        f"warmup={WARMUP_EPOCHS} wd={WEIGHT_DECAY} | "
        f"loss box/cls/dfl={LOSS_BOX}/{LOSS_CLS}/{LOSS_DFL} "
        f"cls_pw={CLS_PW} QFL(β={QFL_BETA}, mix={QFL_BCE_MIX})"
    )


def apply_cli_overrides(args) -> None:
    """用 CLI / CONFIG 覆盖本模块全局超参（就地修改）。"""
    global QFL_BETA, QFL_BCE_MIX, FL_GAMMA, FL_ALPHA
    global LOSS_BOX, LOSS_CLS, LOSS_DFL, CLS_PW
    global OPTIMIZER, LR0, LRF, COS_LR, MOMENTUM, WEIGHT_DECAY
    global WARMUP_EPOCHS, WARMUP_MOMENTUM, WARMUP_BIAS_LR

    if getattr(args, "fl_gamma", None) is not None:
        QFL_BETA = float(args.fl_gamma)
    if getattr(args, "fl_alpha", None) is not None:
        QFL_BCE_MIX = float(args.fl_alpha)
    FL_GAMMA, FL_ALPHA = QFL_BETA, QFL_BCE_MIX

    if getattr(args, "box", None) is not None:
        LOSS_BOX = float(args.box)
    if getattr(args, "cls", None) is not None:
        LOSS_CLS = float(args.cls)
    if getattr(args, "dfl", None) is not None:
        LOSS_DFL = float(args.dfl)
    if getattr(args, "cls_pw", None) is not None:
        CLS_PW = float(args.cls_pw)

    if getattr(args, "optimizer", None):
        OPTIMIZER = str(args.optimizer)
    if getattr(args, "lr0", None) is not None:
        LR0 = float(args.lr0)
    if getattr(args, "lrf", None) is not None:
        LRF = float(args.lrf)
    if getattr(args, "cos_lr", None) is not None:
        COS_LR = bool(args.cos_lr)
    if getattr(args, "momentum", None) is not None:
        MOMENTUM = float(args.momentum)
    if getattr(args, "weight_decay", None) is not None:
        WEIGHT_DECAY = float(args.weight_decay)
    if getattr(args, "warmup_epochs", None) is not None:
        WARMUP_EPOCHS = float(args.warmup_epochs)
