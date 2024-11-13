import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from torch.utils.data import DataLoader

from train import WaymoLoader, pytorch_neg_multi_log_likelihood_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--n-samples", type=int, required=False, default=50)
    parser.add_argument("--use-top1", action="store_true")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    model = torch.jit.load(args.model).cuda().eval()
    loader = DataLoader(
        WaymoLoader(args.data, return_vector=True),
        batch_size=1,
        num_workers=1,
        shuffle=False,
    )

    iii = 0
    with torch.no_grad():
        for x, y, is_available, vector_data in loader:
            x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))

            confidences_logits, logits = model(x)

            argmax = confidences_logits.argmax()
            if args.use_top1:
                confidences_logits = confidences_logits[:, argmax].unsqueeze(1)
                logits = logits[:, argmax].unsqueeze(1)

            loss = pytorch_neg_multi_log_likelihood_batch(
                y, logits, confidences_logits, is_available
            )
            confidences = torch.softmax(confidences_logits, dim=1)
            V = vector_data[0]

            X, idx = V[:, :44], V[:, 44].flatten()

            figure(figsize=(15, 15), dpi=80)
            for i in np.unique(idx):
                _X = X[idx == i]
                if _X[:, 5:12].sum() > 0:
                    plt.plot(_X[:, 0], _X[:, 1], linewidth=4, color="red")
                else:
                    plt.plot(_X[:, 0], _X[:, 1], color="black")
                plt.xlim([-224 // 4, 224 // 4])
                plt.ylim([-224 // 4, 224 // 4])

            logits = logits.squeeze(0).cpu().numpy()
            y = y.squeeze(0).cpu().numpy()
            is_available = is_available.squeeze(0).long().cpu().numpy()
            confidences = confidences.squeeze(0).cpu().numpy()
            plt.plot(
                y[is_available > 0][::10, 0],
                y[is_available > 0][::10, 1],
                "-o",
                label="gt",
            )

            plt.plot(
                logits[confidences.argmax()][is_available > 0][::10, 0],
                logits[confidences.argmax()][is_available > 0][::10, 1],
                "-o",
                label="pred top 1",
            )
            if not args.use_top1:
                for traj_id in range(len(logits)):
                    if traj_id == argmax:
                        continue

                    alpha = confidences[traj_id].item()
                    alpha = 0.7
                    plt.plot(
                        logits[traj_id][is_available > 0][::10, 0],
                        logits[traj_id][is_available > 0][::10, 1],
                        "-o",
                        # label=f"pred {traj_id} {alpha:.3f}",
                        label=f"pred {traj_id}",
                        alpha=alpha,
                    )


            plt.title(loss.item())
            plt.legend()
            plt.savefig(
                os.path.join(args.save, f"{iii:0>2}_{loss.item():.3f}.png")
            )
            plt.close()
            # confidences_logits, logits = model(x)
            # print("Predicted Trajectories:", logits)
            # print("Confidences:", confidences_logits)
            # # import numpy as np
            #
            # # 自定义的距离阈值
            # miss_threshold = 2.0  # 例如2米为miss rate的阈值
            #
            # # 初始化计算指标的容器
            # min_ade = []
            # min_fde = []
            # miss_rate_count = 0
            # total_trajectories = 0
            # overlap_count = 0
            # total_points = 0
            #
            # # 遍历每个场景的预测结果
            # for i in range(len(logits)):
            #     pred_trajectories = logits[i]  # 预测的轨迹
            #     gt_trajectory = y[i]  # 真实轨迹
            #
            #     # 计算 MinADE 和 MinFDE
            #     ade_list = []
            #     fde_list = []
            #
            #     for pred in pred_trajectories:
            #         # 计算ADE (平均位移误差)
            #         ade = np.mean(np.sqrt(np.sum((pred - gt_trajectory) ** 2, axis=1)))
            #         ade_list.append(ade)
            #
            #         # 计算FDE (终点位移误差)
            #         fde = np.sqrt(np.sum((pred[-1] - gt_trajectory[-1]) ** 2))
            #         fde_list.append(fde)
            #
            #     # 记录最小ADE和最小FDE
            #     min_ade.append(min(ade_list))
            #     min_fde.append(min(fde_list))
            #
            #     # 计算Miss Rate (如果最小FDE大于阈值，则计为miss)
            #     if min(fde_list) > miss_threshold:
            #         miss_rate_count += 1
            #
            #     total_trajectories += 1
            #
            #     # 计算Overlap Rate (简单示例：预测点与真实点在阈值范围内计为重叠)
            #     for j in range(len(pred_trajectories[0])):  # 遍历每个时间步的点
            #         pred_points = np.array([pred[j] for pred in pred_trajectories])
            #         gt_point = gt_trajectory[j]
            #
            #         # 判断是否有轨迹点与真实点接近，计为重叠
            #         overlaps = np.sqrt(np.sum((pred_points - gt_point) ** 2, axis=1)) < miss_threshold
            #         if np.any(overlaps):
            #             overlap_count += 1
            #         total_points += 1
            #
            # # 计算Miss Rate和Overlap Rate
            # miss_rate = miss_rate_count / total_trajectories
            # overlap_rate = overlap_count / total_points
            #
            # # 输出结果
            # print(f"MinADE: {np.mean(min_ade):.4f}")
            # print(f"MinFDE: {np.mean(min_fde):.4f}")
            # print(f"Miss Rate: {miss_rate:.4f}")
            # print(f"Overlap Rate: {overlap_rate:.4f}")

            iii += 1
            if iii == args.n_samples:
                break


if __name__ == "__main__":
    main()