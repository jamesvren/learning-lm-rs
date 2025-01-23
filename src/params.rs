use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        //todo!("实现从safetensors文件的模型参数加载");
        let tensors = safetensor.tensors();
        for (name, tensor) in tensors {
            println!("{name} => {:?} {:?}", tensor.dtype(), tensor.shape());
        }
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor_view = safetensor.tensor(name).unwrap();
            if tensor_view.dtype() != safetensors::Dtype::F32 {
                panic!("Data type {:?} is not support", tensor_view.dtype());
            }

            let data: Vec<f32> = tensor_view.data()
                .chunks_exact(4)
                .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            let shape: Vec<usize> = tensor_view.shape().to_vec();

            Tensor::<f32>::new(data, &shape)
        };

        let mut wq: Vec<Tensor<f32>> = Vec::new();
        let mut wk: Vec<Tensor<f32>> = Vec::new();
        let mut wv: Vec<Tensor<f32>> = Vec::new();
        let mut wo: Vec<Tensor<f32>> = Vec::new();
        let mut w_gate: Vec<Tensor<f32>> = Vec::new();
        let mut w_up: Vec<Tensor<f32>> = Vec::new();
        let mut w_down: Vec<Tensor<f32>> = Vec::new();
        let mut rms_att_w: Vec<Tensor<f32>> = Vec::new();
        let mut rms_ffn_w: Vec<Tensor<f32>> = Vec::new();

        for n in 0..config.num_hidden_layers {
            wq.push(get_tensor(&format!("model.layers.{n}.self_attn.q_proj.weight")));
            wk.push(get_tensor(&format!("model.layers.{n}.self_attn.k_proj.weight")));
            wv.push(get_tensor(&format!("model.layers.{n}.self_attn.v_proj.weight")));
            wo.push(get_tensor(&format!("model.layers.{n}.self_attn.o_proj.weight")));
            w_gate.push(get_tensor(&format!("model.layers.{n}.mlp.gate_proj.weight")));
            w_up.push(get_tensor(&format!("model.layers.{n}.mlp.up_proj.weight")));
            w_down.push(get_tensor(&format!("model.layers.{n}.mlp.down_proj.weight")));
            rms_att_w.push(get_tensor(&format!("model.layers.{n}.input_layernorm.weight")));
            rms_ffn_w.push(get_tensor(&format!("model.layers.{n}.post_attention_layernorm.weight")));
        }

        // Iter can not assure sequence of name, so have to use hard code above.
        //
        //for (name, _) in safetensor.iter() {
        //    println!("got {name}");
        //    match name {
        //        n if n.contains("q_proj") => wq.push(get_tensor(n)),
        //        n if n.contains("k_proj") => wk.push(get_tensor(n)),
        //        n if n.contains("v_proj") => wv.push(get_tensor(n)),
        //        n if n.contains("o_proj") => wo.push(get_tensor(n)),
        //        n if n.contains("gate_proj") => w_gate.push(get_tensor(n)),
        //        n if n.contains("up_proj") => w_up.push(get_tensor(n)),
        //        n if n.contains("down_proj") => w_down.push(get_tensor(n)),
        //        n if n.contains("input_layernorm") => rms_att_w.push(get_tensor(n)),
        //        n if n.contains("post_attention_layernorm") => rms_ffn_w.push(get_tensor(n)),
        //        _ => (),
        //    }
        //}

        let lm_head = get_tensor("lm_head.weight");

        let embedding_table = if config.tie_word_embeddings {
            get_tensor("lm_head.weight")
        } else {
            Tensor::<f32>::default(&lm_head.shape())
        };

        Self {
            embedding_table,
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head,
        }
    }
}
