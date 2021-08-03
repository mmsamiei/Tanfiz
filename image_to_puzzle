x = torch.randn(100, 3, 32, 32)  # batch, c, h, w
kh, kw = 8, 8  # kernel size
dh, dw = 8, 8  # stride
patches = x.unfold(2, kh, dh).unfold(3, kw, dw)
batch_size = x.shape[0]
patches = patches.reshape(batch_size,3,16,8,8)

r=torch.randperm(16)
patches = patches[:,:,r,:,:]

def jigsaw_to_image(x, grid_size=(2, 2)):
    # x shape is batch_size x num_patches x c x jigsaw_h x jigsaw_w
    batch_size, num_patches, c, jigsaw_h, jigsaw_w = x.size()
    assert num_patches == grid_size[0] * grid_size[1]
    x_image = x.view(batch_size, grid_size[0], grid_size[1], c, jigsaw_h, jigsaw_w)
    output_h = grid_size[0] * jigsaw_h
    output_w = grid_size[1] * jigsaw_w
    x_image = x_image.permute(0, 3, 1, 4, 2, 5).contiguous()
    x_image = x_image.view(batch_size, c, output_h, output_w)
    return x_image
    
patches = patches.permute(0,2,1,3,4)
puzzle = jigsaw_to_image(patches, grid_size=(4, 4)) ##grid size is num of patches
