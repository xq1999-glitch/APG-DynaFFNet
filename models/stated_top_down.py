from mmpose.models.detectors import TopDown
from mmpose.models.builder import POSENETS

@POSENETS.register_module()   # 相当于一个大框架
class StatedTopDown(TopDown):
    def __init__(self,           # 任意数量的位置参数和关键字参数
                 with_meta=False,
                 *args, **kwargs,):
        super().__init__(*args, **kwargs)
        self.with_meta = with_meta   # 类的实例变量
        
    def forward_train(self, img, target, target_weight, img_metas, **kwargs):   # 输入图像，目标，目标权重，图像元数据，任意数量的关键字参数
        """Defines the computation performed at every call when training."""
        output = self.backbone(img)   # 主干网络
        if self.with_neck:
            output = self.neck(output)   # 有Neck（一般就是特征融合模块）传入neck
        if self.with_keypoint:    # 如果配置关键点检测头，进行
            output = self.keypoint_head(output)    # 得到的是热力图

        # if return loss
        losses = dict()   # 存储损失
        if self.with_keypoint:    # 关键点检测头
            if self.with_meta:     # 如果有元数据，就调用调用keypoint_head的get_loss方法来计算损失，并传递图像元数据；否则，只传递输出、目标和目标权重。
                keypoint_losses = self.keypoint_head.get_loss(
                    output, target, target_weight, img_metas=img_metas)
            else:
                keypoint_losses = self.keypoint_head.get_loss(
                    output, target, target_weight)
            losses.update(keypoint_losses)   # 将计算出的损失更新到losses字典中
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output, target, target_weight)    # 计算关键点检测的准确性
            losses.update(keypoint_accuracy)    # 更新准确性
        # print("topdown里面的哦")
        # print(losses)
        return losses
    
    def set_train_epoch(self, epoch: int):    # 把轮次传给每个组件
        setattr(self.backbone, "train_epoch", epoch)
        if self.with_neck:
            setattr(self.neck, "train_epoch", epoch)
        if self.with_keypoint:
            setattr(self.keypoint_head, "train_epoch", epoch)