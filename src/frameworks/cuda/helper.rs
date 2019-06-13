//! Provides useful macros for easier NN implementation for CUDA/cuDNN.

/// Returns cuDNN ready memory pointer from a SharedTensor.
pub unsafe fn receive_memory_ptr<T>(x: &::co::tensor::SharedTensor<T>, device: &::co::device::DeviceType) -> Result<*const ::libc::c_void, ::co::plugin::Error> {
    Ok(::std::mem::transmute::<u64, *const ::libc::c_void>(
        *x.get(device)
            .ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory."))?
            .as_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive CUDA memory."))?
            .id_c()
    ))
}

/// Returns mutable cuDNN ready memory pointer from a SharedTensor.
pub unsafe fn receive_memory_ptr_mut<T>(x: &mut ::co::tensor::SharedTensor<T>, device: &::co::device::DeviceType) -> Result<*mut ::libc::c_void, ::co::plugin::Error> {
    Ok(::std::mem::transmute::<u64, *mut ::libc::c_void>(
        *x.get_mut(device)
            .ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory."))?
            .as_mut_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive CUDA memory."))?
            .id_c()
    ))
}

// #[macro_export]
macro_rules! impl_oconf_for_cc(($($t: ident), +) => (
    $(
        impl<'a> NNOperationConfig<$t> for utils::ConvolutionConfig { }
    )+
));

// #[macro_export]
macro_rules! impl_oconf_for_clrn(($($t: ident), +) => (
    $(
        impl NNOperationConfig<$t> for utils::NormalizationConfig { }
    )+
));

// #[macro_export]
macro_rules! impl_oconf_for_pooling(($($t: ident), +) => (
    $(
        impl NNOperationConfig<$t> for utils::PoolingConfig { }
    )+
));

macro_rules! impl_ops_sigmoid_for {
    ($t:ident, $b:ty) => (
        impl $crate::plugin::Sigmoid<$t> for $b {
            fn sigmoid(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match result.add_device(self.device()) { _ => () }

                self.sigmoid_plain(x, result)
            }

            fn sigmoid_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();
                let acti = CUDNN.init_activation().unwrap();

                Ok(match CUDNN.sigmoid_forward(
                    &acti,
                    &x.cudnn_tensor_desc_flat()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &result.cudnn_tensor_desc_flat()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }?, // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation Sigmoid Forward."))
                    }
                }?)
            }

            fn sigmoid_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match x_diff.add_device(self.device()) { _ => x.sync(self.device())? }
                match result.add_device(self.device()) { _ => x.sync(self.device())? }
                match result_diff.add_device(self.device()) { _ => () }

                self.sigmoid_grad_plain(x, x_diff, result, result_diff)
            }

            fn sigmoid_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();
                let acti = CUDNN.init_activation().unwrap();

                Ok(match CUDNN.sigmoid_backward(
                    &acti,
                    &x.cudnn_tensor_desc_flat()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &x_diff.cudnn_tensor_desc_flat()?, // src_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }?, //src_diff_data
                    &result.cudnn_tensor_desc_flat()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(result, self.device()) }?, // dest_data
                    &result_diff.cudnn_tensor_desc_flat()?, // dest_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }?, // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation Sigmoid Backward."))
                    }
                }?)
            }
        }
    )
}

// #[macro_export]
macro_rules! impl_ops_sigmoid_pointwise_for {
    ($t:ident, $b:ty) => (
        impl $crate::plugin::SigmoidPointwise<$t> for $b {
            fn sigmoid_pointwise(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }

                self.sigmoid_pointwise_plain(x)
            }

            fn sigmoid_pointwise_plain(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();
                let acti = CUDNN.init_activation().unwrap();

                Ok(match CUDNN.sigmoid_forward(
                    &acti,
                    &x.cudnn_tensor_desc_flat()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &x.cudnn_tensor_desc_flat()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(x, self.device()) }?, // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Sigmoid Pointwise forward."))
                    }
                }?)
            }

            fn sigmoid_pointwise_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match x_diff.add_device(self.device()) { _ => x.sync(self.device())? }

                self.sigmoid_pointwise_grad_plain(x, x_diff)
            }

            fn sigmoid_pointwise_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();
                let acti = CUDNN.init_activation().unwrap();

                Ok(match CUDNN.sigmoid_backward(
                    &acti, 
                    &x.cudnn_tensor_desc_flat()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &x_diff.cudnn_tensor_desc_flat()?, // src_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }?, //src_diff_data
                    &x.cudnn_tensor_desc_flat()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, // dest_data
                    &x_diff.cudnn_tensor_desc_flat()?, // dest_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(x_diff, self.device()) }?, // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Sigmoid Pointwise backward."))
                    }
                }?)
            }
        }
    )
}

// #[macro_export]
macro_rules! impl_ops_relu_for {
    ($t:ident, $b:ty) => (
        impl $crate::plugin::Relu<$t> for $b {
            fn relu(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match result.add_device(self.device()) { _ => () }

                self.relu_plain(x, result)
            }

            fn relu_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();
                let acti = CUDNN.init_activation().unwrap();

                Ok(match CUDNN.relu_forward(
                    &acti,
                    &x.cudnn_tensor_desc_flat()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &result.cudnn_tensor_desc_flat()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }?, // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation relu Forward."))
                    }
                }?)
            }

            fn relu_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match x_diff.add_device(self.device()) { _ => x.sync(self.device())? }
                match result.add_device(self.device()) { _ => x.sync(self.device())? }
                match result_diff.add_device(self.device()) { _ => () }

                self.relu_grad_plain(x, x_diff, result, result_diff)
            }

            fn relu_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();
                let acti = CUDNN.init_activation().unwrap();

                Ok(match CUDNN.relu_backward(
                    &acti,
                    &x.cudnn_tensor_desc_flat()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &x_diff.cudnn_tensor_desc_flat()?, // src_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }?, //src_diff_data
                    &result.cudnn_tensor_desc_flat()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(result, self.device()) }?, // dest_data
                    &result_diff.cudnn_tensor_desc_flat()?, // dest_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }?, // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation relu Backward."))
                    }
                }?)
            }
        }
    )
}

// #[macro_export]
macro_rules! impl_ops_relu_pointwise_for {
    ($t:ident, $b:ty) => (
        impl $crate::plugin::ReluPointwise<$t> for $b {
            fn relu_pointwise(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }

                self.relu_pointwise_plain(x)
            }

            fn relu_pointwise_plain(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();
                let acti = CUDNN.init_activation().unwrap();

                Ok(match CUDNN.relu_forward(
                    &acti,
                    &x.cudnn_tensor_desc_flat()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &x.cudnn_tensor_desc_flat()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(x, self.device()) }?, // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN ReLU Pointwise forward."))
                    }
                }?)
            }

            fn relu_pointwise_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match x_diff.add_device(self.device()) { _ => x.sync(self.device())? }

                self.relu_pointwise_grad_plain(x, x_diff)
            }

            fn relu_pointwise_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();
                let acti = CUDNN.init_activation().unwrap();

                Ok(match CUDNN.relu_backward(
                    &acti,
                    &x.cudnn_tensor_desc_flat()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &x_diff.cudnn_tensor_desc_flat()?, // src_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }?, //src_diff_data
                    &x.cudnn_tensor_desc_flat()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, // dest_data
                    &x_diff.cudnn_tensor_desc_flat()?, // dest_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(x_diff, self.device()) }?, // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN ReLU Pointwise backward."))
                    }
                }?)
            }
        }
    )
}

// #[macro_export]
macro_rules! impl_ops_tanh_for {
    ($t:ident, $b:ty) => (
        impl $crate::plugin::Tanh<$t> for $b {
            fn tanh(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match result.add_device(self.device()) { _ => () }

                self.tanh_plain(x, result)
            }

            fn tanh_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();
                let acti = CUDNN.init_activation().unwrap();

                Ok(match CUDNN.tanh_forward(
                    &acti,
                    &x.cudnn_tensor_desc_flat()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &result.cudnn_tensor_desc_flat()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }?, // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation tanh Forward."))
                    }
                }?)
            }

            fn tanh_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match x_diff.add_device(self.device()) { _ => x.sync(self.device())? }
                match result.add_device(self.device()) { _ => x.sync(self.device())? }
                match result_diff.add_device(self.device()) { _ => () }

                self.tanh_grad_plain(x, x_diff, result, result_diff)
            }

            fn tanh_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();
                let acti = CUDNN.init_activation().unwrap();

                Ok(match CUDNN.tanh_backward(
                    &acti,
                    &x.cudnn_tensor_desc_flat()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &x_diff.cudnn_tensor_desc_flat()?, // src_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }?, //src_diff_data
                    &result.cudnn_tensor_desc_flat()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(result, self.device()) }?, // dest_data
                    &result_diff.cudnn_tensor_desc_flat()?, // dest_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }?, // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation tanh Backward."))
                    }
                }?)
            }
        }
    )
}

// #[macro_export]
macro_rules! impl_ops_tanh_pointwise_for {
    ($t:ident, $b:ty) => (
        impl $crate::plugin::TanhPointwise<$t> for $b {
            fn tanh_pointwise(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }

                self.tanh_pointwise_plain(x)
            }

            fn tanh_pointwise_plain(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();
                let acti = CUDNN.init_activation().unwrap();

                Ok(match CUDNN.tanh_forward(
                    &acti,
                    &x.cudnn_tensor_desc_flat()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &x.cudnn_tensor_desc_flat()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(x, self.device()) }?, // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Tanh Pointwise forward."))
                    }
                }?)
            }

            fn tanh_pointwise_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match x_diff.add_device(self.device()) { _ => x.sync(self.device())? }

                self.tanh_pointwise_grad_plain(x, x_diff)
            }

            fn tanh_pointwise_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();
                let acti = CUDNN.init_activation().unwrap();

                Ok(match CUDNN.tanh_backward(
                    &acti,
                    &x.cudnn_tensor_desc_flat()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &x_diff.cudnn_tensor_desc_flat()?, // src_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }?, //src_diff_data
                    &x.cudnn_tensor_desc_flat()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, // dest_data
                    &x_diff.cudnn_tensor_desc_flat()?, // dest_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(x_diff, self.device()) }?, // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Tanh Pointwise backward."))
                    }
                }?)
            }
        }
    )
}

// #[macro_export]
macro_rules! impl_ops_convolution_for {
    ($t:ty, $b:ty) => (
        fn convolution(
            &self,
            filter: &mut ::co::tensor::SharedTensor<$t>,
            x: &mut ::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            workspace: &mut ::co::tensor::SharedTensor<u8>,
            config: &Self::CC //$crate::frameworks::cuda::CC
        ) -> Result<(), ::co::error::Error> {
            match x.add_device(self.device()) { _ => x.sync(self.device())? }
            match result.add_device(self.device()) { _ => () }
            match workspace.add_device(self.device()) { _ => () }

            self.convolution_plain(filter, x, result, workspace, config)
        }

        fn convolution_plain(
            &self,
            filter: &::co::tensor::SharedTensor<$t>,
            x: &::co::tensor::SharedTensor<$t>,
            result: &mut ::co::tensor::SharedTensor<$t>,
            workspace: &mut ::co::tensor::SharedTensor<u8>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(match CUDNN.convolution_forward(
                config,
                unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(workspace, self.device()) }?,
                unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(filter, self.device()) }?,
                &x.cudnn_tensor_desc()?, // src_desc
                unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                &result.cudnn_tensor_desc()?, // dest_desc
                unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }?, // dest_data
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(_) => {
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation convolution Forward."))
                }
            }?)
        }

        #[allow(unused_variables)]
        fn convolution_grad_filter(
            &self,
            src_data: &mut ::co::tensor::SharedTensor<$t>,
            dest_diff: &mut ::co::tensor::SharedTensor<$t>,
            filter_diff: &mut ::co::tensor::SharedTensor<$t>,
            workspace: &mut ::co::tensor::SharedTensor<u8>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            match src_data.add_device(self.device()) { _ => src_data.sync(self.device())? }
            match dest_diff.add_device(self.device()) { _ => dest_diff.sync(self.device())? }
            match filter_diff.add_device(self.device()) { _ => filter_diff.sync(self.device())? }
            match workspace.add_device(self.device()) { _ => () }

            self.convolution_grad_filter_plain(src_data, dest_diff, filter_diff, workspace, config)
        }

        #[allow(unused_variables)]
        fn convolution_grad_filter_plain(
            &self,
            src_data: &::co::tensor::SharedTensor<$t>,
            dest_diff: &::co::tensor::SharedTensor<$t>,
            filter_diff: &mut ::co::tensor::SharedTensor<$t>,
            workspace: &mut ::co::tensor::SharedTensor<u8>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(match CUDNN.convolution_backward_filter(
                config,
                unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(workspace, self.device()) }?,
                &src_data.cudnn_tensor_desc()?,
                unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(src_data, self.device()) }?,
                &dest_diff.cudnn_tensor_desc()?,
                unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(dest_diff, self.device()) }?,
                unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(filter_diff, self.device()) }?,
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(_) => {
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation convolution Backward."))
                }
            }?)
        }

        #[allow(unused_variables)]
        fn convolution_grad_data(
            &self,
            filter: &mut ::co::tensor::SharedTensor<$t>,
            x_diff: &mut ::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>,
            workspace: &mut ::co::tensor::SharedTensor<u8>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            match filter.add_device(self.device()) { _ => filter.sync(self.device())? }
            match x_diff.add_device(self.device()) { _ => x_diff.sync(self.device())? }
            match result_diff.add_device(self.device()) { _ => result_diff.sync(self.device())? }
            match workspace.add_device(self.device()) { _ => () }

            self.convolution_grad_data_plain(filter, x_diff, result_diff, workspace, config)
        }

        #[allow(unused_variables)]
        fn convolution_grad_data_plain(
            &self,
            filter: &::co::tensor::SharedTensor<$t>,
            x_diff: &::co::tensor::SharedTensor<$t>,
            result_diff: &mut ::co::tensor::SharedTensor<$t>,
            workspace: &mut ::co::tensor::SharedTensor<u8>,
            config: &Self::CC
        ) -> Result<(), ::co::error::Error> {
            let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

            Ok(match CUDNN.convolution_backward_data(
                config,
                unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(workspace, self.device()) }?,
                unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(filter, self.device()) }?,
                &x_diff.cudnn_tensor_desc()?,
                unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }?,
                &result_diff.cudnn_tensor_desc()?,
                unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }?,
                scal_params
            ) {
                Ok(_) => Ok(()),
                Err(_) => {
                    Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation convolution Backward."))
                }
            }?)
        }
    )
}

// #[macro_export]
macro_rules! impl_ops_softmax_for {
    ($t:ident, $b:ty) => (
        impl $crate::plugin::Softmax<$t> for $b {
            fn softmax(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match result.add_device(self.device()) { _ => () }

                self.softmax_plain(x, result)
            }

            fn softmax_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(match CUDNN.softmax_forward(
                    &x.cudnn_tensor_desc_softmax()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &result.cudnn_tensor_desc_softmax()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }?, // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN softmax Forward."))
                    }
                }?)
            }

            fn softmax_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match x_diff.add_device(self.device()) { _ => x.sync(self.device())? }
                match result_diff.add_device(self.device()) { _ => () }

                self.softmax_grad_plain(x, x_diff, result_diff)
            }

            fn softmax_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(match CUDNN.softmax_backward(
                    &x.cudnn_tensor_desc_softmax()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &x_diff.cudnn_tensor_desc_softmax()?, // src_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }?, //src_diff_data
                    &result_diff.cudnn_tensor_desc_softmax()?, // dest_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }?, // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN softmax Backward."))
                    }
                }?)
            }
        }
    )
}

// #[macro_export]
macro_rules! impl_ops_log_softmax_for {
    ($t:ident, $b:ty) => (
        impl $crate::plugin::LogSoftmax<$t> for $b {
            fn log_softmax(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match result.add_device(self.device()) { _ => () }

                self.log_softmax_plain(x, result)
            }

            fn log_softmax_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(match CUDNN.log_softmax_forward(
                    &x.cudnn_tensor_desc_softmax()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &result.cudnn_tensor_desc_softmax()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }?, // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN logarithmic softmax Forward."))
                    }
                }?)
            }

            fn log_softmax_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match x_diff.add_device(self.device()) { _ => x.sync(self.device())? }
                match result_diff.add_device(self.device()) { _ => () }

                self.log_softmax_grad_plain(x, x_diff, result_diff)
            }

            fn log_softmax_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(match CUDNN.log_softmax_backward(
                    &x.cudnn_tensor_desc_softmax()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &x_diff.cudnn_tensor_desc_softmax()?, // src_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }?, //src_diff_data
                    &result_diff.cudnn_tensor_desc_softmax()?, // dest_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }?, // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN logarithmic softmax Backward."))
                    }
                }?)
            }
        }
    )
}

// #[macro_export]
macro_rules! impl_ops_lrn_for {
    ($t:ident, $b:ty) => (
        impl $crate::plugin::LRN<$t> for $b {
            fn new_lrn_config(
                &self,
                n: u32,
                alpha: f64,
                beta: f64,
                k: f64
            ) -> Result<Self::CLRN, ::co::error::Error> {
                Ok(CUDNN.init_normalization(n, alpha, beta, k).unwrap())
            }

            fn lrn(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CLRN //$crate::frameworks::cuda::CC
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match result.add_device(self.device()) { _ => () }

                self.lrn_plain(x, result, config)
            }

            fn lrn_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CLRN
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(match CUDNN.lrn_forward(
                    config,
                    &x.cudnn_tensor_desc()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &result.cudnn_tensor_desc()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }?, // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation lrn Forward."))
                    }
                }?)
            }

            #[allow(unused_variables)]
            fn lrn_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CLRN
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match x_diff.add_device(self.device()) { _ => x.sync(self.device())? }
                match result.add_device(self.device()) { _ => x.sync(self.device())? }
                match result_diff.add_device(self.device()) { _ => () }

                self.lrn_grad_plain(x, x_diff, result, result_diff, config)
            }

            #[allow(unused_variables)]
            fn lrn_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CLRN
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(match CUDNN.lrn_backward(
                    config,
                    &x.cudnn_tensor_desc()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &x_diff.cudnn_tensor_desc()?, // src_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }?, //src_diff_data
                    &result.cudnn_tensor_desc()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(result, self.device()) }?, // dest_data
                    &result_diff.cudnn_tensor_desc()?, // dest_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }?, // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation lrn Backward."))
                    }
                }?)
            }
        }
    )
}

//#[macro_export]
macro_rules! impl_ops_pooling_for {
    ($t:ident, $b:ty) => (
        impl $crate::plugin::Pooling<$t> for $b {
            fn new_pooling_config(
                &self,
                window: &[i32],
                padding: &[i32],
                stride: &[i32],
            ) -> Result<Self::CPOOL, ::co::error::Error> {
                let pooling_avg = ::cudnn::PoolingDescriptor::new(::cudnn::cudnnPoolingMode_t_CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, window, padding, stride).unwrap();
                let pooling_max = ::cudnn::PoolingDescriptor::new(::cudnn::cudnnPoolingMode_t_CUDNN_POOLING_MAX, window, padding, stride).unwrap();
                Ok(::cudnn::utils::PoolingConfig::new(pooling_avg, pooling_max))
            }

            fn pooling_max(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CPOOL
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match result.add_device(self.device()) { _ => () }

                self.pooling_max_plain(x, result, config)
            }

            fn pooling_max_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CPOOL
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(match CUDNN.pooling_max_forward(
                    config,
                    &x.cudnn_tensor_desc()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &result.cudnn_tensor_desc()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result, self.device()) }?, // dest_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation pooling Forward."))
                    }
                }?)
            }

            #[allow(unused_variables)]
            fn pooling_max_grad(
                &self,
                x: &mut ::co::tensor::SharedTensor<$t>,
                x_diff: &mut ::co::tensor::SharedTensor<$t>,
                result: &mut ::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CPOOL
            ) -> Result<(), ::co::error::Error> {
                match x.add_device(self.device()) { _ => x.sync(self.device())? }
                match x_diff.add_device(self.device()) { _ => x.sync(self.device())? }
                match result.add_device(self.device()) { _ => x.sync(self.device())? }
                match result_diff.add_device(self.device()) { _ => () }

                self.pooling_max_grad_plain(x, x_diff, result, result_diff, config)
            }

            #[allow(unused_variables)]
            fn pooling_max_grad_plain(
                &self,
                x: &::co::tensor::SharedTensor<$t>,
                x_diff: &::co::tensor::SharedTensor<$t>,
                result: &::co::tensor::SharedTensor<$t>,
                result_diff: &mut ::co::tensor::SharedTensor<$t>,
                config: &Self::CPOOL
            ) -> Result<(), ::co::error::Error> {
                let scal_params: ::cudnn::utils::ScalParams<$t> = ::cudnn::utils::ScalParams::default();

                Ok(match CUDNN.pooling_max_backward(
                    config,
                    &x.cudnn_tensor_desc()?, // src_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x, self.device()) }?, //src_data
                    &x_diff.cudnn_tensor_desc()?, // src_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(x_diff, self.device()) }?, //src_diff_data
                    &result.cudnn_tensor_desc()?, // dest_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr(result, self.device()) }?, // dest_data
                    &result_diff.cudnn_tensor_desc()?, // dest_diff_desc
                    unsafe { $crate::frameworks::cuda::helper::receive_memory_ptr_mut(result_diff, self.device()) }?, // dest_diff_data
                    scal_params
                ) {
                    Ok(_) => Ok(()),
                    Err(_) => {
                        Err(::co::plugin::Error::Operation("Unable to execute CUDA cuDNN Activation pooling Backward."))
                    }
                }?)
            }
        }
    )
}
