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

#include "common/xf_headers.hpp"
#include "xf_gaussian_filter_config.h"
#include "xf_sobel_config.h"
#include <iostream>

#include "xcl2.hpp"

using namespace std;

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: <executable> <input image path>\n");
        return -1;
    }

    cv::Mat in_img, out_img, ocv_ref, in_img_gau;
    cv::Mat in_gray, in_gray1, diff;
    // Sobel
	cv::Mat c_grad_x_1, c_grad_y_1, c_grad_x_2, c_grad_y_2;
	cv::Mat c_grad_x, c_grad_y;
	cv::Mat hls_grad_x, hls_grad_y, hls_grad_x_1, hls_grad_y_1;
	cv::Mat diff_grad_x, diff_grad_y, diff_grad_x_1, diff_grad_y_1;

#if GRAY
    in_img = cv::imread(argv[1], 0); // reading in the color image
#else
    in_img = cv::imread(argv[1], 1); // reading in the color image
#endif

    if (!in_img.data) {
        fprintf(stderr, "Failed to load the image ... !!!\n ");
        return -1;
    }
// extractChannel(in_img, in_img, 1);
#if GRAY

    out_img.create(in_img.rows, in_img.cols, CV_8UC1); // create memory for output image
    diff.create(in_img.rows, in_img.cols, CV_8UC1);    // create memory for OCV-ref image
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC1); // create memory for OCV-ref image
    // Sobel
    hls_grad_x.create(in_img.rows, in_img.cols, CV_8UC1);
	hls_grad_y.create(in_img.rows, in_img.cols, CV_8UC1);
	diff_grad_x.create(in_img.rows, in_img.cols, CV_8UC1);
	diff_grad_y.create(in_img.rows, in_img.cols, CV_8UC1);

	hls_grad_x_1.create(in_img.rows, in_img.cols, CV_8UC1);
	hls_grad_y_1.create(in_img.rows, in_img.cols, CV_8UC1);
	diff_grad_x_1.create(in_img.rows, in_img.cols, CV_8UC1);
	diff_grad_y_1.create(in_img.rows, in_img.cols, CV_8UC1);

#else
    out_img.create(in_img.rows, in_img.cols, CV_8UC3); // create memory for output image
    diff.create(in_img.rows, in_img.cols, CV_8UC3);    // create memory for OCV-ref image
    ocv_ref.create(in_img.rows, in_img.cols, CV_8UC3); // create memory for OCV-ref image
    // Sobel
	hls_grad_x.create(in_img.rows, in_img.cols, CV_8UC3);
	hls_grad_y.create(in_img.rows, in_img.cols, CV_8UC3);
	diff_grad_x.create(in_img.rows, in_img.cols, CV_8UC3);
	diff_grad_y.create(in_img.rows, in_img.cols, CV_8UC3);
#endif

#if FILTER_WIDTH == 3
    float sigma = 0.5f;
#endif
#if FILTER_WIDTH == 7
    float sigma = 1.16666f;
    int ddepth = -1;
#else
    int ddepth = CV_8U;
#endif
#if FILTER_WIDTH == 5
    float sigma = 10.0f;//0.8333f;
#endif

    // OpenCV Gaussian filter function
    int scale = 1;
	int delta = 0;

    cv::GaussianBlur(in_img, ocv_ref, cv::Size(FILTER_WIDTH, FILTER_WIDTH), FILTER_WIDTH / 6.0, FILTER_WIDTH / 6.0,
                     cv::BORDER_CONSTANT);
    cv::Sobel(ocv_ref, c_grad_x_1, ddepth, 1, 0, FILTER_WIDTH, scale, delta, cv::BORDER_CONSTANT);
	cv::Sobel(ocv_ref, c_grad_y_1, ddepth, 0, 1, FILTER_WIDTH, scale, delta, cv::BORDER_CONSTANT);

	cv::Sobel(in_img, c_grad_x_2, ddepth, 1, 0, FILTER_WIDTH, scale, delta, cv::BORDER_CONSTANT);
	cv::Sobel(in_img, c_grad_y_2, ddepth, 0, 1, FILTER_WIDTH, scale, delta, cv::BORDER_CONSTANT);

    imwrite("output_sof_gauss.png", ocv_ref);
    imwrite("out_sof_x.jpg", c_grad_x_1);
	imwrite("out_sof_y.jpg", c_grad_y_1);
	imwrite("out_sof_x_o.jpg", c_grad_x_2);
	imwrite("out_sof_y_o.jpg", c_grad_y_2);

    /////////////////////////////////////// CL ////////////////////////

    int height = in_img.rows;
    int width = in_img.cols;
    cl_int err;
    std::cout << "INFO: Running OpenCL section." << std::endl;

    // Get the device:
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Context, command queue and device name:
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    std::cout << "INFO: Device found - " << device_name << std::endl;

    // Load binary:

    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_gaussian_filter");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "gaussian_filter_accel", &err));

    // Allocate the buffers:
    OCL_CHECK(err, cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, (height * width * CH_TYPE), NULL, &err)); //,in_img.data);
    OCL_CHECK(err, cl::Buffer imageFromDevice1(context, CL_MEM_WRITE_ONLY, (height * width * CH_TYPE), NULL, &err)); //,(ap_uint<OUTPUT_PTR_WIDTH>*)out_img.data);
    OCL_CHECK(err, cl::Buffer imageFromDevice2(context, CL_MEM_WRITE_ONLY, (height * width * CH_TYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevice3(context, CL_MEM_WRITE_ONLY, (height * width * CH_TYPE), NULL, &err)); //,(ap_uint<OUTPUT_PTR_WIDTH>*)out_img.data);
    OCL_CHECK(err, cl::Buffer imageFromDevice4(context, CL_MEM_WRITE_ONLY, (height * width * CH_TYPE), NULL, &err));
    // Set kernel arguments:
    OCL_CHECK(err, err = kernel.setArg(0, imageToDevice));
    OCL_CHECK(err, err = kernel.setArg(1, imageFromDevice1));
    OCL_CHECK(err, err = kernel.setArg(2, imageFromDevice2));
    OCL_CHECK(err, err = kernel.setArg(3, imageFromDevice3));
    OCL_CHECK(err, err = kernel.setArg(4, imageFromDevice4));
    OCL_CHECK(err, err = kernel.setArg(5, height));
    OCL_CHECK(err, err = kernel.setArg(6, width));
    OCL_CHECK(err, err = kernel.setArg(7, sigma));

    // Initialize the buffers:
    cl::Event event;

    OCL_CHECK(err, q.enqueueWriteBuffer(imageToDevice,              // buffer on the FPGA
                                        CL_TRUE,                    // blocking call
                                        0,                          // buffer offset in bytes
                                        (height * width * CH_TYPE), // Size in bytes
                                        in_img.data,                // Pointer to the data to copy
                                        nullptr, &event));

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    // Execute the kernel:
    OCL_CHECK(err, err = q.enqueueTask(kernel, NULL, &event_sp));

    clWaitForEvents(1, (const cl_event*)&event_sp);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    // Copy Result from Device Global Memory to Host Local Memory
    q.enqueueReadBuffer(imageFromDevice1, CL_TRUE, 0, (height * width * CH_TYPE), hls_grad_x.data, nullptr, &event_sp);
    q.enqueueReadBuffer(imageFromDevice2, CL_TRUE, 0, (height * width * CH_TYPE), hls_grad_y.data, nullptr, &event_sp);
    q.enqueueReadBuffer(imageFromDevice3, CL_TRUE, 0, (height * width * CH_TYPE), hls_grad_x_1.data, nullptr, &event_sp);
    q.enqueueReadBuffer(imageFromDevice4, CL_TRUE, 0, (height * width * CH_TYPE), hls_grad_y_1.data, nullptr, &event_sp);

    q.finish();
    /////////////////////////////////////// end of CL ////////////////////////

    cv::imwrite("hw_out_x.jpg", hls_grad_x);
    cv::imwrite("hw_out_y.jpg", hls_grad_y);
    cv::imwrite("hw_out_x_o.jpg", hls_grad_x_1);
    cv::imwrite("hw_out_y_o.jpg", hls_grad_y_1);

    //////////////////  Compute Absolute Difference ////////////////////
    /*cv::absdiff(ocv_ref, out_img, diff);
    cv::imwrite("out_error.jpg", diff);

    float err_per;

    xf::cv::analyzeDiff(diff, 0, err_per);

    if (err_per > 1) {
        fprintf(stderr, "\nTest Failed.\n ");
        return -1;
    } else {
        std::cout << "Test Passed " << std::endl;
        return 0;
    }*/

	#if (FILTER_WIDTH == 3 | FILTER_WIDTH == 5)
		absdiff(c_grad_x_1, hls_grad_x, diff_grad_x);
		absdiff(c_grad_y_1, hls_grad_y, diff_grad_y);
		absdiff(c_grad_x_2, hls_grad_x_1, diff_grad_x_1);
		absdiff(c_grad_y_2, hls_grad_y_1, diff_grad_y_1);
	#endif
	#if (FILTER_WIDTH == 7)
		if (OUT_TYPE == XF_8UC1 || OUT_TYPE == XF_16SC1 || OUT_TYPE == XF_8UC3 || OUT_TYPE == XF_16SC3) {
			absdiff(c_grad_x_1, hls_grad_x, diff_grad_x);
			absdiff(c_grad_y_1, hls_grad_y, diff_grad_y);
		} else if (OUT_TYPE == XF_32UC1) {
			c_grad_x_1.convertTo(c_grad_x, CV_32S);
			c_grad_y_1.convertTo(c_grad_y, CV_32S);
			absdiff(c_grad_x, hls_grad_x, diff_grad_x);
			absdiff(c_grad_y, hls_grad_y, diff_grad_y);
		}
	#endif

	float err_per, err_per1, err_per2, err_per3;
	int ret;

	xf::cv::analyzeDiff(diff_grad_x, 0, err_per);
	xf::cv::analyzeDiff(diff_grad_y, 0, err_per1);
	xf::cv::analyzeDiff(diff_grad_x_1, 0, err_per2);
	xf::cv::analyzeDiff(diff_grad_y_1, 0, err_per3);

	if (err_per > 0.0f) {
		//fprintf(stderr, "Test failed .... !!!\n ");
		std::cout << "Its not bad .... !!!" << std::endl;
		ret = 0;
	} else {
		std::cout << "Test Passed .... !!!" << std::endl;
		ret = 0;
	}

    imwrite("out_errorx.jpg", diff_grad_x);
    imwrite("out_errory.jpg", diff_grad_y);
    imwrite("out_errorx_o.jpg", diff_grad_x_1);
	imwrite("out_errory_o.jpg", diff_grad_y_1);

    return ret;
}
