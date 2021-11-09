import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from util import util_data


def pca(title, data, data_index, data_label, class_num):
    X_pca = PCA(n_components=2).fit_transform(data)
    font = {"color": "darkred", "size": 13, "family": "serif"}
    # plt.style.use("dark_background")
    plt.style.use("default")
    plt.figure()

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data_index, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', class_num))
    if data_label:
        for i in range(len(X_pca)):
            plt.annotate(data_label[i], xy=(X_pca[:, 0][i], X_pca[:, 1][i]),
                         xytext=(X_pca[:, 0][i] + 0.00, X_pca[:, 1][i] + 0.00))
    plt.title(title, fontdict=font)

    if data_label is None:
        cbar = plt.colorbar(ticks=range(class_num))
        # cbar.set_label(label='digit value', fontdict=font)
        plt.clim(0 - 0.5, class_num - 0.5)
    plt.savefig('{}.pdf'.format(title))
    plt.show()


def t_sne(title, data, data_index, data_label, class_num):
    print('processing data')
    X_tsne = TSNE(n_components=2).fit_transform(data)  # [num_samples, n_components]
    print('processing data over')

    font = {"color": "darkred", "size": 13, "family": "serif"}
    # plt.style.use("dark_background")
    plt.style.use("default")

    plt.figure()
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data_index, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', class_num))
    if data_label:
        for i in range(len(X_tsne)):
            plt.annotate(data_label[i], xy=(X_tsne[:, 0][i], X_tsne[:, 1][i]),
                         xytext=(X_tsne[:, 0][i] + 1, X_tsne[:, 1][i] + 1))
    plt.title(title, fontdict=font)

    if data_label is None:
        cbar = plt.colorbar(ticks=range(class_num))
        # cbar.set_label(label='digit value', fontdict=font)
        plt.clim(0 - 0.5, class_num - 0.5)
    plt.savefig('{}.pdf'.format(title))
    plt.show()


def draw_residue_heatmap(std_acid_count, xticklabels, yticklabels, title):
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    f, ax = plt.subplots(figsize=(14, 4))
    plt.subplots_adjust(top=0.85, wspace=0.2, hspace=0.3)
    ax = sns.heatmap(std_acid_count, xticklabels=xticklabels, yticklabels=yticklabels, linewidths=0.5, cmap="YlGnBu", annot=True)
    ax.set_title(title, fontsize=18)
    plt.savefig('../figures/{}.pdf'.format(title))
    plt.show()

    return None


if __name__ == '__main__':
    # conv_feature_maps = torch.load('../main/conv_feature_maps.pt')
    # print('conv_feature_maps', len(conv_feature_maps), conv_feature_maps)
    # for i, feature_map in enumerate(conv_feature_maps):
    #     print('feature_map[{}]: {}'.format(i, feature_map.size()))

    filters_weight = torch.load('../main/filters_weight.pt')
    print('filters_weight', len(filters_weight), filters_weight)

    filter_size_i_vis = None
    for i, filter_size_i in enumerate(filters_weight):
        print('filter_size_i[{}]: {}'.format(i, filter_size_i.size()))
        filter_size_i_vis = filter_size_i

    filter_size_i_vis = filter_size_i_vis.squeeze()
    filter_size_i_vis_sum = torch.zeros([64, 128])
    for filter_j_in_size_i in filter_size_i_vis:
        filter_size_i_vis_sum = filter_size_i_vis_sum + filter_j_in_size_i

    print('filter_size_i_vis_sum', filter_size_i_vis_sum.size())
    filter_size_i_vis_sum = filter_size_i_vis_sum.detach()

    # filter_size_i_vis_sum = torch.sum(filter_size_i_vis_sum, dim=1, keepdim=True)

    # TODO: 可视化卷积核
    f, ax = plt.subplots(figsize=(14, 4))
    # figure, ax = plt.subplots()
    plt.subplots_adjust(top=0.85, left=0.1, right=0.9, wspace=0.2, hspace=0.3)
    cbar_ax = f.add_axes([.94, .1, .011, .748])  # x_location, y_location, width, height
    ax = sns.heatmap(filter_size_i_vis_sum, xticklabels=[i for i in range(filter_size_i_vis_sum.size(1))],
                     yticklabels=[i for i in range(filter_size_i_vis_sum.size(0))],
                     linewidths=0.5, cmap="YlGnBu", cbar_ax=cbar_ax, ax=ax)
    ax.set_title("(A) Residue Propensity in Training Set", fontsize=18)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)

    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    ax.set_xlabel('embedding dimension', fontsize=10)
    ax.set_ylabel('position', fontsize=10)

    # plt.savefig('../figures/filter_weight.pdf')
    # plt.savefig('../figures/filter_weight.pdf')
    plt.show()
