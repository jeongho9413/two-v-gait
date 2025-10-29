import math
import os

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataset_dir = "./datasets/KUGait_VLP32C_2022-Spring-C/train"  # path/to/dataset

    # ortho. 64x44
    train_x_ortho_dep_side = np.load( os.path.join(dataset_dir, "OrthoRes004Size64x44", "train_x_dep_side.npy") )
    train_x_ortho_gei_side = np.load( os.path.join(dataset_dir, "OrthoRes004Size64x44", "train_x_gei_side.npy") )

    train_x_ortho_dep_back = np.load( os.path.join(dataset_dir, "OrthoRes004Size64x44", "train_x_dep_back.npy") )
    train_x_ortho_gei_back = np.load( os.path.join(dataset_dir, "OrthoRes004Size64x44", "train_x_gei_back.npy") )

    train_x_ortho_dep_nongde = np.load( os.path.join(dataset_dir, "OrthoRes004Size64x44", "train_x_dep_nongde.npy") )
    train_x_ortho_gei_nongde = np.load( os.path.join(dataset_dir, "OrthoRes004Size64x44", "train_x_gei_nongde.npy") )

    # cylin 64x44
    train_x_cylin_dep_nongde_nonnorm = np.load( os.path.join(dataset_dir, "CylinSize64x44", "train_x_dep_nongde_nonnorm.npy") )
    train_x_cylin_dep_nongde_norm = np.load( os.path.join(dataset_dir, "CylinSize64x44", "train_x_dep_nongde_norm.npy") )
    train_x_cylin_gei_nongde_norm = np.load( os.path.join(dataset_dir, "CylinSize64x44", "train_x_gei_nongde_norm.npy") )

    train_x_cylin_dep_gde_nonnorm = np.load( os.path.join(dataset_dir, "CylinSize64x44", "train_x_dep_gde_nonnorm.npy") )
    train_x_cylin_dep_gde_norm = np.load( os.path.join(dataset_dir, "CylinSize64x44", "train_x_dep_gde_norm.npy") )
    train_x_cylin_gei_gde_norm = np.load( os.path.join(dataset_dir, "CylinSize64x44", "train_x_gei_gde_norm.npy") )

    B = train_x_ortho_dep_side.shape[0]  # define the batch size and length
    L = train_x_ortho_dep_side.shape[1]

    png_dir = './pngs/'
    os.makedirs(png_dir, exist_ok=True)

    for b in range(B):
        for l in range(L):
            plt.close()
            fig, axes = plt.subplots(5, 3, figsize=(40, 60))

            ax11, ax12, ax13 = axes[0]
            ax21, ax22, ax23 = axes[1]
            ax31, ax32, ax33 = axes[2]
            ax41, ax42, ax43 = axes[3]
            ax51, ax52, ax53 = axes[4]

            ax12.imshow(train_x_ortho_dep_nongde[b][l], cmap=plt.cm.gray, aspect=1)
            ax13.imshow(train_x_ortho_gei_nongde[b][0], cmap=plt.cm.gray, aspect=1)

            ax22.imshow(train_x_ortho_dep_side[b][l], cmap=plt.cm.gray, aspect=1)
            ax23.imshow(train_x_ortho_gei_side[b][0], cmap=plt.cm.gray, aspect=1)

            ax32.imshow(train_x_ortho_dep_back[b][l], cmap=plt.cm.gray, aspect=1)
            ax33.imshow(train_x_ortho_gei_back[b][0], cmap=plt.cm.gray, aspect=1)

            ax41.imshow(train_x_cylin_dep_nongde_nonnorm[b][l], cmap=plt.cm.gray, aspect=1)
            ax42.imshow(train_x_cylin_dep_nongde_norm[b][l], cmap=plt.cm.gray, aspect=1)
            ax43.imshow(train_x_cylin_gei_nongde_norm[b][0], cmap=plt.cm.gray, aspect=1)

            ax51.imshow(train_x_cylin_dep_gde_nonnorm[b][l], cmap=plt.cm.gray, aspect=1)
            ax52.imshow(train_x_cylin_dep_gde_norm[b][l], cmap=plt.cm.gray, aspect=1)
            ax53.imshow(train_x_cylin_gei_gde_norm[b][0], cmap=plt.cm.gray, aspect=1)

            for row in axes:
                for ax in row:
                    ax.axis("off")

            ax12.set_title('Ortho + Non-GDE + Depth', fontsize='40')
            ax13.set_title('Ortho + Non-GDE + GEI', fontsize='40')

            ax22.set_title('Ortho + GDE (Side) + Depth', fontsize='40')
            ax23.set_title('Ortho + GDE (Side) + GEI', fontsize='40')

            ax32.set_title('Ortho + GDE (Back) + Depth', fontsize='40')
            ax33.set_title('Ortho + GDE (Back) + GEI', fontsize='40')

            ax41.set_title('Cylin + Non-GDE + Depth + Non-Norm', fontsize='40')
            ax42.set_title('Cylin + Non-GDE + Depth + Norm', fontsize='40')
            ax43.set_title('Cylin + Non-GDE + GEI + Norm', fontsize='40')

            ax51.set_title('Cylin + GDE (Side) + Depth + Non-Norm', fontsize='40')
            ax52.set_title('Cylin + GDE (Side) + Depth + Norm', fontsize='40')
            ax53.set_title('Cylin + GDE (Side) + GEI + Norm', fontsize='40')

            fig.tight_layout()
            png_name = f"{b:05d}_{l:02d}.png"
            fig.savefig(os.path.join(png_dir, png_name), bbox_inches="tight")
            plt.close(fig)