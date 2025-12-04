import numpy as np
import scipy.linalg
import torch


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160


    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray or torch.Tensor
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
            Can be a single [4] vector or a batch [B, 4].

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
            For batch input, returns [B, 8] and [B, 8, 8].
        """
        # 1. 确保我们处理的是 CPU 上的 numpy 数组
        if isinstance(measurement, torch.Tensor):
            mean_pos = measurement.cpu().numpy()
        else:
            mean_pos = measurement

        # 2. 检查是单个还是一批
        if mean_pos.ndim == 1:
            # 原始的 SOT (单目标) 逻辑
            mean_pos = mean_pos.reshape(1, -1) # 提升到 [1, 4]
            was_batch = False
        else:
            # MOT (多目标) 逻辑
            was_batch = True

        # mean_pos 现在是 [B, 4]
        B = mean_pos.shape[0]
        mean_vel = np.zeros_like(mean_pos) # [B, 4]

        # 3. 沿列 (axis=1) 拼接
        mean = np.concatenate([mean_pos, mean_vel], axis=1) # [B, 8]

        # 4. 创建一批 (B) std 矩阵
        std = np.zeros((B, 8))
        h = mean_pos[:, 3]  # 获取每个 bbox 的高度 [h1, h2, ...]

        # position std
        std[:, 0] = 2 * self._std_weight_position * h
        std[:, 1] = 2 * self._std_weight_position * h
        std[:, 2] = 1e-2
        std[:, 3] = 2 * self._std_weight_position * h

        # velocity std
        std[:, 4] = 10 * self._std_weight_velocity * h
        std[:, 5] = 10 * self._std_weight_velocity * h
        std[:, 6] = 1e-5
        std[:, 7] = 10 * self._std_weight_velocity * h

        std_squared = np.square(std) # [B, 8]

        # 5. 创建一批 (B) [8, 8] 的对角协方差矩阵
        covariance = np.zeros((B, 8, 8))

        # 使用 einsum 进行高效的批量对角矩阵创建
        # 'bi,ij->bij' : b=batch, i=row, j=col
        # 这会将 [B, 8] 转换为 [B, 8, 8]
        covariance = np.einsum('bi,ij->bij', std_squared, np.eye(8))

        if not was_batch:
            # 如果输入是单个，则降维回原始格式
            mean = mean[0]
            covariance = covariance[0]

        return mean, covariance

    # def initiate(self, measurement):
    #     """Create track from unassociated measurement.

    #     Parameters
    #     ----------
    #     measurement : ndarray
    #         Bounding box coordinates (x, y, a, h) with center position (x, y),
    #         aspect ratio a, and height h.

    #     Returns
    #     -------
    #     (ndarray, ndarray)
    #         Returns the mean vector (8 dimensional) and covariance matrix (8x8
    #         dimensional) of the new track. Unobserved velocities are initialized
    #         to 0 mean.

    #     """
    #     mean_pos = measurement
    #     mean_vel = np.zeros_like(mean_pos)
    #     mean = np.r_[mean_pos, mean_vel]

    #     std = [
    #         2 * self._std_weight_position * measurement[3],
    #         2 * self._std_weight_position * measurement[3],
    #         1e-2,
    #         2 * self._std_weight_position * measurement[3],
    #         10 * self._std_weight_velocity * measurement[3],
    #         10 * self._std_weight_velocity * measurement[3],
    #         1e-5,
    #         10 * self._std_weight_velocity * measurement[3]]
    #     covariance = np.diag(np.square(std))
    #     return mean, covariance


    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The [B, 8] mean vector of the object state at the previous time step.
        covariance : ndarray
            The [B, 8, 8] covariance matrix of the object state at the previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the predicted [B, 8] mean vector and [B, 8, 8] covariance matrix.
        """

        # 1. Check if batched or not.
        if mean.ndim == 1:
            # If 1D, lift to 2D [1, 8]
            mean = mean.reshape(1, -1)
            covariance = covariance.reshape(1, 8, 8)
            was_batch = False
        else:
            was_batch = True

        B = mean.shape[0]

        # 2. Create batched motion covariance Q [B, 8, 8]

        # Get height 'h' for all objects. Shape: [B]
        # This is the line that fixes the bug.
        h = mean[:, 3] 

        std_pos = np.zeros((B, 4))
        std_pos[:, 0] = self._std_weight_position * h
        std_pos[:, 1] = self._std_weight_position * h
        std_pos[:, 2] = 1e-2
        std_pos[:, 3] = self._std_weight_position * h

        std_vel = np.zeros((B, 4))
        std_vel[:, 0] = self._std_weight_velocity * h
        std_vel[:, 1] = self._std_weight_velocity * h
        std_vel[:, 2] = 1e-5
        std_vel[:, 3] = self._std_weight_velocity * h

        # Combine std_pos and std_vel. Shape: [B, 8]
        std_squared = np.square(np.concatenate([std_pos, std_vel], axis=1))

        # Create batched diagonal covariance matrix Q. Shape: [B, 8, 8]
        motion_cov = np.einsum('bi,ij->bij', std_squared, np.eye(8))

        # 3. Perform batched prediction
        # F is [8, 8]. mean is [B, 8].
        # We want: mean = (F @ mean.T).T
        mean_pred = np.einsum('ij,bj->bi', self._motion_mat, mean)

        # F is [8, 8]. covariance is [B, 8, 8].
        # We want: cov = F @ cov @ F.T + Q
        # Step 1: F @ cov  -> [B, 8, 8]
        temp_cov = np.einsum('ik,bkj->bij', self._motion_mat, covariance)
        # Step 2: (F @ cov) @ F.T  -> [B, 8, 8]
        cov_pred = np.einsum('bij,kj->bik', temp_cov, self._motion_mat)
        # Step 3: (F @ cov @ F.T) + Q
        cov_pred += motion_cov

        if not was_batch:
            # If input was 1D, return 1D
            mean_pred = mean_pred[0]
            cov_pred = cov_pred[0]

        return mean_pred, cov_pred

    # def predict(self, mean, covariance):
    #     """Run Kalman filter prediction step.

    #     Parameters
    #     ----------
    #     mean : ndarray
    #         The 8 dimensional mean vector of the object state at the previous
    #         time step.
    #     covariance : ndarray
    #         The 8x8 dimensional covariance matrix of the object state at the
    #         previous time step.

    #     Returns
    #     -------
    #     (ndarray, ndarray)
    #         Returns the mean vector and covariance matrix of the predicted
    #         state. Unobserved velocities are initialized to 0 mean.

    #     """
    #     std_pos = [
    #         self._std_weight_position * mean[3],
    #         self._std_weight_position * mean[3],
    #         1e-2,
    #         self._std_weight_position * mean[3]]
    #     std_vel = [
    #         self._std_weight_velocity * mean[3],
    #         self._std_weight_velocity * mean[3],
    #         1e-5,
    #         self._std_weight_velocity * mean[3]]
    #     motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

    #     #mean = np.dot(self._motion_mat, mean)
    #     mean = np.dot(mean, self._motion_mat.T)
    #     covariance = np.linalg.multi_dot((
    #         self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

    #     return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector. Can be [8] (single) or [B, 8] (batch).
        covariance : ndarray
            The state's covariance matrix. Can be [8, 8] (single) or [B, 8, 8] (batch).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """

        # 1. 检查是单个还是一批 (lift to 2D if 1D)
        if mean.ndim == 1:
            mean = mean.reshape(1, -1)
            covariance = covariance.reshape(1, 8, 8)
            was_batch = False
        else:
            was_batch = True

        B = mean.shape[0] # Batch size (B=2 for you)

        # 2. 创建批量的 innovation covariance R [B, 4, 4]

        # 修正 BUG: 从 [B, 8] 中获取所有对象的高度 'h' (第3列)
        # h 的形状将是 [B]
        h = mean[:, 3] 

        std = np.zeros((B, 4))
        std[:, 0] = self._std_weight_position * h
        std[:, 1] = self._std_weight_position * h
        std[:, 2] = 1e-1
        std[:, 3] = self._std_weight_position * h

        std_squared = np.square(std) # [B, 4]

        # 创建批量的对角矩阵 R. 形状: [B, 4, 4]
        innovation_cov = np.einsum('bi,ij->bij', std_squared, np.eye(4))

        # 3. 执行批量投影

        # 投影 mean: H @ mean.T
        # H 是 [4, 8]. mean 是 [B, 8].
        # (H @ mean[b].T) -> [4, 8] @ [8, 1] = [4, 1]
        # 我们使用 einsum: 'ij,bj->bi' -> [4, 8] @ [B, 8] -> [B, 4]
        projected_mean = np.einsum('ij,bj->bi', self._update_mat, mean)

        # 投影 covariance: H @ P @ H.T
        # H 是 [4, 8]. P (covariance) 是 [B, 8, 8]. H.T 是 [8, 4]

        # 步骤 1: H @ P -> [4, 8] @ [B, 8, 8] -> [B, 4, 8]
        temp_cov = np.einsum('ik,bkj->bij', self._update_mat, covariance)

        # 步骤 2: (H @ P) @ H.T -> [B, 4, 8] @ [8, 4] -> [B, 4, 4]
        projected_cov = np.einsum('bij,kj->bik', temp_cov, self._update_mat)

        # 4. 如果输入是1D，则降维回1D
        if not was_batch:
            projected_mean = projected_mean[0]
            projected_cov = projected_cov[0]
            innovation_cov = innovation_cov[0]

        return projected_mean, projected_cov + innovation_cov

    # def project(self, mean, covariance):
    #     """Project state distribution to measurement space.

    #     Parameters
    #     ----------
    #     mean : ndarray
    #         The state's mean vector (8 dimensional array).
    #     covariance : ndarray
    #         The state's covariance matrix (8x8 dimensional).

    #     Returns
    #     -------
    #     (ndarray, ndarray)
    #         Returns the projected mean and covariance matrix of the given state
    #         estimate.

    #     """
    #     std = [
    #         self._std_weight_position * mean[3],
    #         self._std_weight_position * mean[3],
    #         1e-1,
    #         self._std_weight_position * mean[3]]
    #     innovation_cov = np.diag(np.square(std))

    #     mean = np.dot(self._update_mat, mean)
    #     covariance = np.linalg.multi_dot((
    #         self._update_mat, covariance, self._update_mat.T))
    #     return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector. Can be [8] or [B, 8].
        covariance : ndarray
            The state's covariance matrix. Can be [8, 8] or [B, 8, 8].
        measurement : ndarray
            The measurement vector. Can be [4] or [B, 4].

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """

        # 1. 确保是 Numpy 并在需要时提升（lift）维度
        if isinstance(measurement, torch.Tensor):
            measurement = measurement.cpu().numpy()

        if mean.ndim == 1:
            # 输入是单个对象, 提升到 2D 以便统一处理
            mean = mean.reshape(1, -1)
            covariance = covariance.reshape(1, 8, 8)
            measurement = measurement.reshape(1, -1)
            was_batch = False
        else:
            was_batch = True

        B = mean.shape[0] # Batch size (B=2 for you)

        # 2. Project (我们上一步修复的 `project` 函数会返回 [B, 4] 和 [B, 4, 4])
        projected_mean, projected_cov = self.project(mean, covariance)

        # 3. 准备空的列表来收集结果
        new_mean_list = []
        new_covariance_list = []

        # 4. 遍历批次中的每个对象 (B=2)
        #    因为 scipy.linalg 不支持批处理，我们必须循环
        for i in range(B):
            # 提取第 i 个对象的所有数据
            mean_i = mean[i]                     # [8]
            covariance_i = covariance[i]         # [8, 8]
            measurement_i = measurement[i]       # [4]
            projected_mean_i = projected_mean[i] # [4]
            projected_cov_i = projected_cov[i]   # [4, 4] - 这是一个 2D 矩阵！

            # --- 现在，这里是原始的 `update` 逻辑，但只在 2D 矩阵上运行 ---

            # projected_cov_i 现在是 [4, 4]，cho_factor 可以处理它
            # Add regularization to ensure positive definiteness (numerical stability)
            epsilon = 1e-6
            regularized_cov = projected_cov_i + epsilon * np.eye(projected_cov_i.shape[0])
            
            try:
                chol_factor, lower = scipy.linalg.cho_factor(
                    regularized_cov, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                # If still not positive definite, add more regularization
                epsilon = 1e-4
                regularized_cov = projected_cov_i + epsilon * np.eye(projected_cov_i.shape[0])
                chol_factor, lower = scipy.linalg.cho_factor(
                    regularized_cov, lower=True, check_finite=False)

            # b = (P @ H.T).T = H @ P
            b = np.dot(covariance_i, self._update_mat.T).T

            # K = (S^-1 @ (H@P).T).T = (S^-1 @ b).T
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower), b,
                check_finite=False).T

            innovation = measurement_i - projected_mean_i

            new_mean_i = mean_i + np.dot(innovation, kalman_gain.T)
            # Use regularized_cov for consistency (Kalman gain was computed using regularized covariance)
            new_covariance_i = covariance_i - np.linalg.multi_dot((
                kalman_gain, regularized_cov, kalman_gain.T))

            # -----------------------------------------------------------

            # 5. 将结果添加到列表中
            new_mean_list.append(new_mean_i)
            new_covariance_list.append(new_covariance_i)

        # 6. 将结果列表堆叠回 [B, 8] 和 [B, 8, 8]
        new_mean = np.stack(new_mean_list, axis=0)
        new_covariance = np.stack(new_covariance_list, axis=0)

        # 7. 如果输入不是批次，则降维
        if not was_batch:
            new_mean = new_mean[0]
            new_covariance = new_covariance[0]

        return new_mean, new_covariance

    # def update(self, mean, covariance, measurement):
    #     """Run Kalman filter correction step.

    #     Parameters
    #     ----------
    #     mean : ndarray
    #         The predicted state's mean vector (8 dimensional).
    #     covariance : ndarray
    #         The state's covariance matrix (8x8 dimensional).
    #     measurement : ndarray
    #         The 4 dimensional measurement vector (x, y, a, h), where (x, y)
    #         is the center position, a the aspect ratio, and h the height of the
    #         bounding box.

    #     Returns
    #     -------
    #     (ndarray, ndarray)
    #         Returns the measurement-corrected state distribution.

    #     """
    #     projected_mean, projected_cov = self.project(mean, covariance)

    #     chol_factor, lower = scipy.linalg.cho_factor(
    #         projected_cov, lower=True, check_finite=False)
    #     kalman_gain = scipy.linalg.cho_solve(
    #         (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
    #         check_finite=False).T
    #     innovation = measurement - projected_mean

    #     new_mean = mean + np.dot(innovation, kalman_gain.T)
    #     new_covariance = covariance - np.linalg.multi_dot((
    #         kalman_gain, projected_cov, kalman_gain.T))
    #     return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')

    # def compute_iou(self, pred_bbox, bboxes):
    #     """
    #     Compute the IoU between the bbox and the bboxes
    #     """
    #     ious = []
    #     pred_bbox = self.xyah_to_xyxy(pred_bbox)
    #     for bbox in bboxes:
    #         iou = self._compute_iou(pred_bbox, bbox)
    #         ious.append(iou)
    #     return ious
        
    def compute_iou(self, pred_bboxes, bboxes_batch):
        pred_bboxes_xyxy = self.xyah_to_xyxy(pred_bboxes)
        B, M, _ = bboxes_batch.shape
        batched_ious = []
        for i in range(B):
            pred_bbox_i = pred_bboxes_xyxy[i]
            bboxes_m = bboxes_batch[i]     
            ious_for_this_object = []
            for m in range(M):
                bbox_m = bboxes_m[m]
                iou = self._compute_iou(pred_bbox_i, bbox_m)
                ious_for_this_object.append(iou)
            batched_ious.append(ious_for_this_object)
        return batched_ious

    def _compute_iou(self, bbox1, bbox2):
        """
        Compute the Intersection over Union (IoU) of two bounding boxes.
        Parameters
        ----------
        bbox1 : list
            The first bounding box in the format [x1, y1, x2, y2].
        bbox2 : list
            The second bounding box in the format [x1, y1, x2, y2].
        Returns
        -------
        float
            The IoU of the two bounding boxes.
        """
        if bbox2 == [0, 0, 0, 0]:
            return 0
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2
        # Calculate intersection area
        intersection_area = max(0, min(x2, x2_) - max(x1, x1_)) * max(0, min(y2, y2_) - max(y1, y1_))
        # Calculate union area
        union_area = (x2 - x1) * (y2 - y1) + (x2_ - x1_) * (y2_ - y1_) - intersection_area
        # Calculate IoU
        iou = intersection_area / union_area if union_area != 0 else 0
        return iou

    # def xyxy_to_xyah(self, bbox):
    #     x1, y1, x2, y2 = bbox
    #     xc = (x1 + x2) / 2
    #     yc = (y1 + y2) / 2
    #     w = x2 - x1
    #     h = y2 - y1
    #     if h == 0:
    #         h = 1
    #     return [xc, yc, w / h, h]
    
    def xyxy_to_xyah(self, bboxes):
        # 确保输入是 tensor
        if not isinstance(bboxes, torch.Tensor):
            bboxes = torch.tensor(bboxes, device=self.mean.device)

        # 检查输入是 1D (单个 bbox) 还是 2D (一批 bboxes)
        if bboxes.dim() == 1:
            # 原始的非批次行为
            x1, y1, x2, y2 = bboxes
        else:
            # 新的、能感知批次的行为
            x1 = bboxes[:, 0]
            y1 = bboxes[:, 1]
            x2 = bboxes[:, 2]
            y2 = bboxes[:, 3]

        w = x2 - x1
        h = y2 - y1
        x = x1 + 0.5 * w
        y = y1 + 0.5 * h

        # 沿最后一个维度堆叠
        if bboxes.dim() == 1:
            return torch.stack([x, y, w, h], dim=0)
        else:
            return torch.stack([x, y, w, h], dim=1)

    # def xyah_to_xyxy(self, bbox):
    #     xc, yc, a, h = bbox
    #     x1 = xc - a * h / 2
    #     y1 = yc - h / 2
    #     x2 = xc + a * h / 2
    #     y2 = yc + h / 2
    #     return [x1, y1, x2, y2]

    def xyah_to_xyxy(self, bboxes):
        # 确保我们处理的是 numpy 数组
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()

        # 检查是 1D (单个 bbox) 还是 2D (一批 bboxes)
        if bboxes.ndim == 1:
            # 原始的非批次行为
            xc, yc, a, h = bboxes
            x1 = xc - a * h / 2
            y1 = yc - h / 2
            x2 = xc + a * h / 2
            y2 = yc + h / 2
            # 原始函数返回一个 list，我们保持一致
            return [x1, y1, x2, y2]
        else:
            # 新的、能感知批次的行为
            xc = bboxes[:, 0]
            yc = bboxes[:, 1]
            a  = bboxes[:, 2]
            h  = bboxes[:, 3]

            x1 = xc - a * h / 2
            y1 = yc - h / 2
            x2 = xc + a * h / 2
            y2 = yc + h / 2

            # 堆叠回一个 [B, 4] 的数组
            return np.stack([x1, y1, x2, y2], axis=1)
