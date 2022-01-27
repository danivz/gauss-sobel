/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "xf_gaussian_filter_config.h"
#include "xf_sobel_config.h"
#include "imgproc/xf_duplicateimage.hpp"

extern "C" {
void gaussian_filter_accel(
    ap_uint<INPUT_PTR_WIDTH>* img_inp,
	ap_uint<OUTPUT_PTR_WIDTH>* img_out1,
	ap_uint<OUTPUT_PTR_WIDTH>* img_out2,
	ap_uint<OUTPUT_PTR_WIDTH>* img_out3,
	ap_uint<OUTPUT_PTR_WIDTH>* img_out4,
	int rows,
	int cols,
	float sigma)
{
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_inp   offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_out1  offset=slave bundle=gmem2
	#pragma HLS INTERFACE m_axi     port=img_out2  offset=slave bundle=gmem3
	#pragma HLS INTERFACE m_axi     port=img_out3  offset=slave bundle=gmem4
	#pragma HLS INTERFACE m_axi     port=img_out4  offset=slave bundle=gmem5
    #pragma HLS INTERFACE s_axilite port=sigma     
    #pragma HLS INTERFACE s_axilite port=rows     
    #pragma HLS INTERFACE s_axilite port=cols     
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> in_mat(rows, cols);
    // clang-format off
    #pragma HLS stream variable=in_mat.data depth=2
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> in_mat1(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> in_mat2(rows, cols);
    // clang-format off
	#pragma HLS stream variable=in_mat1.data depth=2
	#pragma HLS stream variable=in_mat2.data depth=2
    // clang-format on
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> out_mat(rows, cols);
	#pragma HLS stream variable=out_mat.data depth=2

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> _dstgx(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> _dstgy(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> _dstgx_1(rows, cols);
	xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> _dstgy_1(rows, cols);
	#pragma HLS stream variable=_dstgx.data depth=2
	#pragma HLS stream variable=_dstgy.data depth=2
	#pragma HLS stream variable=_dstgx_1.data depth=2
	#pragma HLS stream variable=_dstgy_1.data depth=2
// clang-format off

// clang-format on

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(img_inp, in_mat);
    xf::cv::duplicateMat<TYPE, HEIGHT, WIDTH, NPC1>(in_mat, in_mat1, in_mat2);

    xf::cv::GaussianBlur<FILTER_WIDTH, XF_BORDER_CONSTANT, TYPE, HEIGHT, WIDTH, NPC1>(in_mat1, out_mat, sigma);

    xf::cv::Sobel<XF_BORDER_CONSTANT, FILTER_WIDTH, TYPE, TYPE, HEIGHT, WIDTH, NPC1, XF_USE_URAM>(out_mat, _dstgx, _dstgy);
    xf::cv::Sobel<XF_BORDER_CONSTANT, FILTER_WIDTH, TYPE, TYPE, HEIGHT, WIDTH, NPC1, XF_USE_URAM>(in_mat2, _dstgx_1, _dstgy_1);

    //xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(out_mat, img_out);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(_dstgx, img_out1);
	xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(_dstgy, img_out2);
	xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(_dstgx_1, img_out3);
	xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(_dstgy_1, img_out4);
}
}
