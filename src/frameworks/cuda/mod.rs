//! Provides NN for a CUDA backend.
#![allow(missing_docs)]
use crate::plugin::*;

use co::prelude::*;
use co::plugin::Error as PluginError;
use cudnn::*;

#[macro_use]
pub mod helper;

lazy_static! {
    static ref CUDNN: Cudnn = Cudnn::new().unwrap();
}

pub trait ICudnnDesc<T> {
    fn cudnn_tensor_desc(&self) -> Result<TensorDescriptor, PluginError>;
    /// Creates a TensorDescriptor similar to `cudnn_tensor_desc`,
    /// but will create a fitting 4D tensor if the actual tensor would be 1D-3D.
    fn cudnn_tensor_desc_softmax(&self) -> Result<TensorDescriptor, PluginError>;
    /// Creates a TensorDescriptor similar to `cudnn_tensor_desc`,
    /// but will create a fitting 3D tensor if the actual tensor would be 1D/2D.
    ///
    /// This should be used in operations where the shape doesn't really matter
    /// e.g. activation like ReLU.
    fn cudnn_tensor_desc_flat(&self) -> Result<TensorDescriptor, PluginError>;

    fn cudnn_filter_desc(&self) -> Result<FilterDescriptor, PluginError>;

    fn cudnn_convolution_desc(&self, filter: &SharedTensor<T>) -> Result<ConvolutionDescriptor, PluginError>;
}

macro_rules! impl_icudnndesc_for_sharedtensor {
    ($t:ty, $cutype:path) => (
        impl ICudnnDesc<$t> for SharedTensor<$t> {
            fn cudnn_tensor_desc(&self) -> Result<TensorDescriptor, PluginError> {
                info!("{:?}", self.desc());

                let td = TensorDescriptor::new(
                    &self.desc().dims_i32().clone(), 
                    &self.desc().default_stride_i32().clone(), 
                    $cutype
                );

                match td {
                    Ok(desc) => Ok(desc),
                    Err(_) => {
                        Err(PluginError::Plugin("Unable to create CuDNN TensorDescriptor."))
                    }
                }
            }

            fn cudnn_tensor_desc_softmax(&self) -> Result<TensorDescriptor, PluginError> {
                let actual_desc = self.desc().clone();
                let override_desc = match actual_desc.len() {
                    // not batched and single dimension softmax
                    1 => vec![1, actual_desc[0], 1, 1],
                    // batched and single dimension softmax
                    2 => vec![actual_desc[0], actual_desc[1], 1, 1],
                    // neither batched nor single dimension
                    3 => vec![1, actual_desc[0], actual_desc[1], actual_desc[2]],
                    _ => actual_desc
                };

                match TensorDescriptor::new(&override_desc.dims_i32().clone(),
                                            &override_desc.default_stride_i32().clone(),
                                            $cutype) {
                    Ok(desc) => Ok(desc),
                    Err(_) => {
                        Err(PluginError::Plugin("Unable to create CuDNN TensorDescriptor."))
                    }
                }
            }

            fn cudnn_tensor_desc_flat(&self) -> Result<TensorDescriptor, PluginError> {
                let actual_desc = self.desc().clone();
                let mut override_desc = match actual_desc.len() {
                    1 => vec![1, 1],
                    2 => vec![1],
                    _ => vec![]
                };
                for dim in actual_desc {
                    override_desc.push(dim);
                }
                match TensorDescriptor::new(&override_desc.dims_i32().clone(),
                                            &override_desc.default_stride_i32().clone(),
                                            $cutype) {
                    Ok(desc) => Ok(desc),
                    Err(_) => {
                        Err(PluginError::Plugin("Unable to create CuDNN TensorDescriptor."))
                    }
                }
            }

            fn cudnn_filter_desc(&self) -> Result<FilterDescriptor, PluginError> {
                match FilterDescriptor::new(&self.desc().dims_i32().clone(), $cutype) {
                    Ok(desc) => Ok(desc),
                    Err(_) => {
                        Err(PluginError::Plugin("Unable to create CuDNN FilterDescriptor."))
                    }
                }
            }

            fn cudnn_convolution_desc(&self, filter: &SharedTensor<$t>) -> Result<ConvolutionDescriptor, PluginError> {
                match ConvolutionDescriptor::new(&self.desc().dims_i32().clone(), &filter.desc().default_stride_i32().clone(), $cutype) {
                    Ok(desc) => Ok(desc),
                    Err(_) => {
                        Err(PluginError::Plugin("Unable to create CuDNN ConvolutionDescriptor."))
                    }
                }
            }
        }
    )
}

impl_icudnndesc_for_sharedtensor!(f32, ::cudnn::utils::DataType::Float);
impl_icudnndesc_for_sharedtensor!(f64, ::cudnn::utils::DataType::Double);

impl_oconf_for_cc!(f32, f64);
impl_oconf_for_clrn!(f32, f64);
impl_oconf_for_pooling!(f32, f64);

impl ConvForwardAlgo {
    /// Tries to return the matching cuDNN type for the enum value.
    fn as_cudnn(&self) -> Result<cudnnConvolutionFwdAlgo_t, ::co::error::Error> {
        Ok(match *self {
            ConvForwardAlgo::Auto => return Err(::co::error::Error::Plugin(::co::plugin::Error::Plugin("Can't create cuDNN convolution forward algorithm from ConvForwardAlgo::Auto. Use `find_cudnn_algo` to find an algorithm."))),
            ConvForwardAlgo::GEMM => ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            ConvForwardAlgo::ImplicitGEMM => ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            ConvForwardAlgo::ImplicitPrecompiledGEMM => ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            ConvForwardAlgo::FFT => ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_FFT,
            ConvForwardAlgo::FFTTiling => ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
            ConvForwardAlgo::Direct => ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
            ConvForwardAlgo::Winograd => ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
            ConvForwardAlgo::WinogradNonFused => ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        })
    }

    /// Returns the matching enum value for a cuDNN algo.
    fn from_cudnn(algo: &cudnnConvolutionFwdAlgo_t) -> ConvForwardAlgo {
        match *algo {
            ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_GEMM => ConvForwardAlgo::GEMM,
            ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => ConvForwardAlgo::ImplicitGEMM,
            ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM => ConvForwardAlgo::ImplicitPrecompiledGEMM,
            ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_FFT => ConvForwardAlgo::FFT,
            ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING => ConvForwardAlgo::FFTTiling,
            ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_DIRECT => ConvForwardAlgo::Direct,
            ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD => ConvForwardAlgo::Winograd,
            ::cudnn::cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED => ConvForwardAlgo::WinogradNonFused,
            _ => unreachable!()
        }
    }

    /// Try to find best algorithm for a operation that uses the provided descriptors.
    fn find_cudnn_algo(
        &self,
        filter_desc: &FilterDescriptor,
        conv_desc: &ConvolutionDescriptor,
        src_desc: &TensorDescriptor,
        dest_desc: &TensorDescriptor,
    ) -> Result<ConvForwardAlgo, ::co::error::Error> {
        if !self.is_auto() {
            return Ok(*self);
        }
        let algos = API::find_convolution_forward_algorithm(*CUDNN.id_c(), *filter_desc.id_c(), *conv_desc.id_c(), *src_desc.id_c(), *dest_desc.id_c()).unwrap();
        let algo = match algos.len() {
            0 => return Err(::co::error::Error::Plugin(::co::plugin::Error::Operation("Unable to find CUDA cuDNN convolution forward algorithm."))),
            _ => algos[0].algo
        };
        Ok(ConvForwardAlgo::from_cudnn(&algo))
    }
}

impl ConvBackwardFilterAlgo {
    /// Tries to return the matching cuDNN type for the enum value.
    fn as_cudnn(&self) -> Result<cudnnConvolutionBwdFilterAlgo_t, ::co::error::Error> {
        Ok(match *self {
            ConvBackwardFilterAlgo::Auto => return Err(::co::error::Error::Plugin(::co::plugin::Error::Plugin("Can't create cuDNN convolution backward filter algorithm from ConvBackwardFilterAlgo::Auto. Use `find_cudnn_algo` to find an algorithm."))),
            ConvBackwardFilterAlgo::ImplicitGEMM => ::cudnn::cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
            ConvBackwardFilterAlgo::ImplicitGEMMSum => ::cudnn::cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            ConvBackwardFilterAlgo::ImplicitPrecompiledGEMMSum => ::cudnn::cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
            ConvBackwardFilterAlgo::FFT => ::cudnn::cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
            ConvBackwardFilterAlgo::FFTTiling => ::cudnn::cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
            ConvBackwardFilterAlgo::WinogradNonFused => ::cudnn::cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
        })
    }

    /// Returns the matching enum value for a cuDNN algo.
    fn from_cudnn(algo: &cudnnConvolutionBwdFilterAlgo_t) -> ConvBackwardFilterAlgo {
        match *algo {
            ::cudnn::cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 => ConvBackwardFilterAlgo::ImplicitGEMMSum,
            ::cudnn::cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 => ConvBackwardFilterAlgo::ImplicitGEMM,
            ::cudnn::cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT => ConvBackwardFilterAlgo::FFT,
            ::cudnn::cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 => ConvBackwardFilterAlgo::ImplicitPrecompiledGEMMSum,
            ::cudnn::cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING => ConvBackwardFilterAlgo::FFTTiling,
            ::cudnn::cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED => ConvBackwardFilterAlgo::WinogradNonFused,
            _ => unreachable!()
        }
    }

    /// Try to find best algorithm for a operation that uses the provided descriptors.
    fn find_cudnn_algo(
        &self,
        filter_desc: &FilterDescriptor,
        conv_desc: &ConvolutionDescriptor,
        src_desc: &TensorDescriptor,
        dest_desc: &TensorDescriptor,
    ) -> Result<ConvBackwardFilterAlgo, ::co::error::Error> {
        if !self.is_auto() {
            return Ok(*self);
        }
        let algos = API::find_convolution_backward_filter_algorithm(*CUDNN.id_c(), *filter_desc.id_c(), *conv_desc.id_c(), *src_desc.id_c(), *dest_desc.id_c()).unwrap();
        let algo = match algos.len() {
            0 => return Err(::co::error::Error::Plugin(::co::plugin::Error::Operation("Unable to find CUDA cuDNN convolution backward filter algorithm."))),
            _ => algos[0].algo
        };
        Ok(ConvBackwardFilterAlgo::from_cudnn(&algo))
    }
}

impl ConvBackwardDataAlgo {
    /// Tries to return the matching cuDNN type for the enum value.
    fn as_cudnn(&self) -> Result<cudnnConvolutionBwdDataAlgo_t, ::co::error::Error> {
        Ok(match *self {
            ConvBackwardDataAlgo::Auto => return Err(::co::error::Error::Plugin(::co::plugin::Error::Plugin("Can't create cuDNN convolution backward data algorithm from ConvBackwardDataAlgo::Auto. Use `find_cudnn_algo` to find an algorithm."))),
            ConvBackwardDataAlgo::ImplicitGEMM => ::cudnn::cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
            ConvBackwardDataAlgo::ImplicitGEMMSum => ::cudnn::cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            ConvBackwardDataAlgo::FFT => ::cudnn::cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
            ConvBackwardDataAlgo::FFTTiling => ::cudnn::cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
            ConvBackwardDataAlgo::Winograd => ::cudnn::cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
            ConvBackwardDataAlgo::WinogradNonFused => ::cudnn::cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED,
        })
    }

    /// Returns the matching enum value for a cuDNN algo.
    fn from_cudnn(algo: &cudnnConvolutionBwdDataAlgo_t) -> ConvBackwardDataAlgo {
        match *algo {
            ::cudnn::cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 => ConvBackwardDataAlgo::ImplicitGEMMSum,
            ::cudnn::cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 => ConvBackwardDataAlgo::ImplicitGEMM,
            ::cudnn::cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT => ConvBackwardDataAlgo::FFT,
            ::cudnn::cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING => ConvBackwardDataAlgo::FFTTiling,
            ::cudnn::cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD => ConvBackwardDataAlgo::Winograd,
            ::cudnn::cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED => ConvBackwardDataAlgo::WinogradNonFused,
            _ => unreachable!()
        }
    }

    /// Try to find best algorithm for a operation that uses the provided descriptors.
    fn find_cudnn_algo(
        &self,
        filter_desc: &FilterDescriptor,
        conv_desc: &ConvolutionDescriptor,
        src_desc: &TensorDescriptor,
        dest_desc: &TensorDescriptor,
    ) -> Result<ConvBackwardDataAlgo, ::co::error::Error> {
        if !self.is_auto() {
            return Ok(*self);
        }
        let algos = API::find_convolution_backward_data_algorithm(*CUDNN.id_c(), *filter_desc.id_c(), *conv_desc.id_c(), *src_desc.id_c(), *dest_desc.id_c()).unwrap();
        let algo = match algos.len() {
            0 => return Err(::co::error::Error::Plugin(::co::plugin::Error::Operation("Unable to find CUDA cuDNN convolution backward data algorithm."))),
            _ => algos[0].algo
        };
        Ok(ConvBackwardDataAlgo::from_cudnn(&algo))
    }
}

macro_rules! impl_convolution_for_cuda_backend {
    ($t:ty, $cutype:path) => (
        impl ConvolutionConfig<$t> for ::cudnn::utils::ConvolutionConfig {
            fn workspace_size(&self) -> usize {
                *self.largest_workspace_size()
            }
        }

        impl Convolution<$t> for Backend<Cuda> {
            fn new_convolution_config(
                &self,
                src: &::co::tensor::SharedTensor<$t>,
                dest: &::co::tensor::SharedTensor<$t>,
                filter: &mut ::co::tensor::SharedTensor<$t>,
                algo_fwd: ConvForwardAlgo,
                algo_bwd_filter: ConvBackwardFilterAlgo,
                algo_bwd_data: ConvBackwardDataAlgo,
                stride: &[i32],
                zero_padding: &[i32],
            ) -> Result<Self::CC, ::co::error::Error> {
                let src_desc = src.cudnn_tensor_desc()?;
                let dest_desc = dest.cudnn_tensor_desc()?;
                let filter_desc = filter.cudnn_filter_desc()?;

                let conv_desc = ::cudnn::ConvolutionDescriptor::new(zero_padding, stride, $cutype).unwrap();

                info!("1");
                let useable_algo_fwd = algo_fwd.find_cudnn_algo(&filter_desc, &conv_desc, &src_desc, &dest_desc)?;

                info!("2");
                let useable_algo_bwd_filter = algo_bwd_filter.find_cudnn_algo(&filter_desc, &conv_desc, &src_desc, &dest_desc)?;
                
                info!("3");
                let useable_algo_bwd_data = algo_bwd_data.find_cudnn_algo(&filter_desc, &conv_desc, &src_desc, &dest_desc)?;

                info!("4");
                let mut workspace_size_fwd = API::get_convolution_forward_workspace_size(*CUDNN.id_c(), useable_algo_fwd.as_cudnn().unwrap(), *filter_desc.id_c(), *conv_desc.id_c(), *src_desc.id_c(), *dest_desc.id_c()).unwrap();
                info!("5");
                let mut workspace_size_bwd_filter = API::get_convolution_backward_filter_workspace_size(*CUDNN.id_c(), useable_algo_bwd_filter.as_cudnn().unwrap(), *filter_desc.id_c(), *conv_desc.id_c(), *src_desc.id_c(), *dest_desc.id_c()).unwrap();
                info!("6");
                let mut workspace_size_bwd_data = API::get_convolution_backward_data_workspace_size(*CUDNN.id_c(), useable_algo_bwd_data.as_cudnn().unwrap(), *filter_desc.id_c(), *conv_desc.id_c(), *src_desc.id_c(), *dest_desc.id_c()).unwrap();

                if workspace_size_fwd == 0 {
                    workspace_size_fwd = 8;
                }

                if workspace_size_bwd_filter == 0 {
                    workspace_size_bwd_filter = 8;
                }
                
                if workspace_size_bwd_data == 0 {
                    workspace_size_bwd_data = 8;
                }

                Ok(
                    ::cudnn::utils::ConvolutionConfig::new(
                        useable_algo_fwd.as_cudnn().unwrap(), 
                        workspace_size_fwd,
                        useable_algo_bwd_filter.as_cudnn().unwrap(), 
                        workspace_size_bwd_filter,
                        useable_algo_bwd_data.as_cudnn().unwrap(), 
                        workspace_size_bwd_data,
                        conv_desc, filter_desc
                    )
                )
            }

            impl_ops_convolution_for!($t, Backend<Cuda>);
        }
    )
}

impl NN<f32> for Backend<Cuda> {
    type CC = utils::ConvolutionConfig;
    type CLRN = utils::NormalizationConfig;
    type CPOOL = utils::PoolingConfig;

    fn init_nn() { let _ = CUDNN.id_c(); }
    fn device(&self) -> &DeviceType { self.device() }
}

impl_convolution_for_cuda_backend!(f32, ::cudnn::utils::DataType::Float);
impl_ops_sigmoid_for!(f32, Backend<Cuda>);
impl_ops_relu_for!(f32, Backend<Cuda>);
impl_ops_tanh_for!(f32, Backend<Cuda>);
impl_ops_softmax_for!(f32, Backend<Cuda>);
impl_ops_log_softmax_for!(f32, Backend<Cuda>);
impl_ops_lrn_for!(f32, Backend<Cuda>);
impl_ops_pooling_for!(f32, Backend<Cuda>);

impl_ops_sigmoid_pointwise_for!(f32, Backend<Cuda>);
impl_ops_relu_pointwise_for!(f32, Backend<Cuda>);
impl_ops_tanh_pointwise_for!(f32, Backend<Cuda>);

impl NN<f64> for Backend<Cuda> {
    type CC = utils::ConvolutionConfig;
    type CLRN = utils::NormalizationConfig;
    type CPOOL = utils::PoolingConfig;

    fn init_nn() { let _ = CUDNN.id_c(); }
    fn device(&self) -> &DeviceType { self.device() }
}

impl_convolution_for_cuda_backend!(f64, ::cudnn::utils::DataType::Double);
impl_ops_sigmoid_for!(f64, Backend<Cuda>);
impl_ops_relu_for!(f64, Backend<Cuda>);
impl_ops_tanh_for!(f64, Backend<Cuda>);
impl_ops_softmax_for!(f64, Backend<Cuda>);
impl_ops_log_softmax_for!(f64, Backend<Cuda>);
impl_ops_lrn_for!(f64, Backend<Cuda>);
impl_ops_pooling_for!(f64, Backend<Cuda>);

impl_ops_sigmoid_pointwise_for!(f64, Backend<Cuda>);
impl_ops_relu_pointwise_for!(f64, Backend<Cuda>);
impl_ops_tanh_pointwise_for!(f64, Backend<Cuda>);
