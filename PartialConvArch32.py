import torch.nn as nn 
import torch


class Partial_Conv(nn.Module):
    def __init__(self, input_filters, output_filters, kernel_size=(3,3), padding=0, stride=1, bias=True, groups=1,dilation=1):
        super().__init__()
        padding = (kernel_size[0]//2, kernel_size[1]//2)
        self.input_conv = nn.Conv2d(input_filters,output_filters,kernel_size,stride,padding,dilation,groups,bias)
        self.mask_conv = nn.Conv2d(input_filters,output_filters,kernel_size,stride,padding,dilation,groups, False)

        self.window_size = kernel_size[0]*kernel_size[1]
        nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False
    
    def forward(self, input, mask):
        output = self.input_conv(input * mask)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        mask_ratio = self.window_size / (output_mask + 1e-8)
        output_mask = torch.clamp(output_mask, 0, 1)
        mask_ratio = mask_ratio * output_mask
        output = output * mask_ratio

        return output, output_mask
    
class EncoderLayer(nn.Module):
    def __init__(self, in_filters, output_filters):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = output_filters
        self.conv_1 = Partial_Conv(self.in_filters, self.out_filters, stride=1)
        self.act_1 = nn.ReLU()
        self.conv_2 = Partial_Conv(self.out_filters, self.out_filters, stride=2)
        self.act_2 = nn.ReLU()

    def forward(self, inputs, masks):
        conv_out_1, mask_out_1 = self.conv_1(inputs, masks)
        conv_out_1 = self.act_1(conv_out_1)
        conv_out_2, mask_out_2 = self.conv_2(conv_out_1, mask_out_1)
        conv_out_2 = self.act_2(conv_out_2)
        return conv_out_1, mask_out_1, conv_out_2, mask_out_2
    
class DecoderLayer(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.up_image = nn.Upsample(scale_factor=(2,2))
        self.up_mask = nn.Upsample(scale_factor=(2,2))

        self.conv_1 = Partial_Conv(self.in_filters*2, self.in_filters)
        self.act_1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv_2 = Partial_Conv(self.in_filters, self.out_filters)
        self.act_2 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input_image, input_mask, shared_image, shared_mask):
        up_image, up_mask = self.up_image(input_image), self.up_mask(input_mask)

        concat_image = torch.cat((shared_image, up_image), axis=1)
        concat_mask = torch.cat((shared_mask, up_mask), axis=1)

        conv_out_1, mask_out_1 = self.conv_1(concat_image, concat_mask)
        conv_out_1 = self.act_1(conv_out_1)
        conv_out_2, mask_out_2 = self.conv_2(conv_out_1, mask_out_1)
        conv_out_2 = self.act_2(conv_out_2)
        return conv_out_1, mask_out_1, conv_out_2, mask_out_2
    
class InpaintingModel(nn.Module):
    """
    Inputs: Image tensor in form C*H*W and Mask tensor in form 1*H*W tensor in 
            Default parameters are 3*32*32

    Assertion: image size should be 32*32 
    
    Output: Image tensor in form C*H*W

    Mask: Tensor values are 1 for removing, 255 for preserving parts of the image
    """
    def __init__(self, num_channels=3):
        super().__init__()
        self.encoder_1 = EncoderLayer(num_channels, 32)
        self.encoder_2 = EncoderLayer(32, 64)
        self.encoder_3 = EncoderLayer(64, 128)
        self.encoder_4 = EncoderLayer(128, 256)

        self.decoder_1 = DecoderLayer(256, 128)
        self.decoder_2 = DecoderLayer(128, 64)
        self.decoder_3 = DecoderLayer(64, 32)
        self.decoder_4 = DecoderLayer(32, 3)

        self.output_layer = nn.Sequential(
            nn.Conv2d(3, 3, (3, 3), padding=1),
            nn.Sigmoid()
        )

    def forward(self, inputs, masks):
        #input: 32*32 with 3 channels
        conv_1, mask_1, conv_2, mask_2 = self.encoder_1(inputs, masks)
        #conv_1: 32*32 with 32 ch. conv_2: 16*16 with 32 ch.
        conv_3, mask_3, conv_4, mask_4 = self.encoder_2(conv_2, mask_2)
        #conv_3: 16*16 with 64 ch. conv_4: 8*8 with 64 ch.
        conv_5, mask_5, conv_6, mask_6 = self.encoder_3(conv_4, mask_4)
        #conv_5: 8*8 with 128 ch. conv_6: 4*4 with 128 ch.
        conv_7, mask_7, conv_8, mask_8 = self.encoder_4(conv_6, mask_6)
        #conv_7: 4*4 with 256 ch. conv_8: 2*2 with 256 ch.

        conv_9, mask_9, conv_10, mask_10 = self.decoder_1(conv_8, mask_8, conv_7, mask_7)
        #conv_9: 4*4 with 256 ch. #conv_10: 4*4 with 128 ch.
        conv_11, mask_11, conv_12, mask_12 = self.decoder_2(conv_10, mask_10, conv_5, mask_5)
        #conv_11: 8*8 with 128 ch. #conv_12: 8*8 with 64 ch.
        conv_13, mask_13, conv_14, mask_14 = self.decoder_3(conv_12, mask_12, conv_3, mask_3)
        #conv_13: 16*16 with 64 ch. #conv_14: 16*16 with 32 ch.
        conv_15, mask_15, conv_16, mask_16 = self.decoder_4(conv_14, mask_14, conv_1, mask_1)
        #conv_15: 32*32 with 32 ch. #conv_16: 32*32 with 3 ch.

        outputs = self.output_layer(conv_16)
        return outputs