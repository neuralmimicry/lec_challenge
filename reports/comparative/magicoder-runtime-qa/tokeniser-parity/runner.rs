//! Vocabulary-only FFI diagnostic against the exact pinned llama.h ABI.
//! No transformer tensors, context, generation or GPU allocation is created.
use std::{error::Error, ffi::{c_char,c_void,CString}, fs, path::Path};
use libloading::{Library,Symbol};
use serde_json::json;
type Result<T> = std::result::Result<T,Box<dyn Error>>;
#[repr(C)]#[derive(Clone,Copy)]union OverrideValue{integer:i64,number:f64,boolean:bool,string:[c_char;128]}
#[repr(C)]#[derive(Clone,Copy)]struct Override{tag:i32,key:[c_char;128],value:OverrideValue}
#[repr(C)]struct Params{devices:*mut *mut c_void,tensor_buft_overrides:*const c_void,n_gpu_layers:i32,split_mode:i32,main_gpu:i32,tensor_split:*const f32,progress_callback:Option<unsafe extern "C" fn(f32,*mut c_void)->bool>,progress_callback_user_data:*mut c_void,kv_overrides:*const Override,vocab_only:bool,use_mmap:bool,use_direct_io:bool,use_mlock:bool,check_tensors:bool,use_extra_bufts:bool,no_host:bool,no_alloc:bool}
fn chars(s:&str)->[c_char;128]{let mut a=[0;128];assert!(s.len()<128);for (v,b)in a.iter_mut().zip(s.bytes()){*v=b as c_char}a}
fn main()->Result<()>{
    let out=Path::new("reports/comparative/magicoder-runtime-qa/tokeniser-parity");fs::create_dir_all(out)?;
    if out.join("report.json").exists(){return Err("preserve existing tokeniser evidence".into())}
    fs::copy("/tmp/lec-magicoder-token-parity.rs",out.join("runner.rs"))?;
    let prompt=fs::read_to_string("reports/comparative/magicoder-runtime-qa/round-002/original-rendered-prompt.txt")?;
    let original=tokenizers::Tokenizer::from_file("offline_bundle/comparative/magicoder-s-ds-6_7b-q4_k_m/source/tokenizer.json").map_err(|e|e.to_string())?;
    let native=original.encode(prompt.as_str(),false).map_err(|e|e.to_string())?.get_ids().to_vec();
    let overrides=[Override{tag:3,key:chars("tokenizer.ggml.model"),value:OverrideValue{string:chars("gpt2")}},Override{tag:3,key:chars("tokenizer.ggml.pre"),value:OverrideValue{string:chars("deepseek-coder")}},Override{tag:2,key:chars("tokenizer.ggml.add_bos_token"),value:OverrideValue{boolean:false}},Override{tag:0,key:[0;128],value:OverrideValue{integer:0}}];
    // SAFETY: symbols and repr(C) structures match the pinned, inspected header
    // retained by the engine receipt. Strings/buffers remain live through calls.
    let (cpp_without_special,cpp_with_special)=unsafe{
        let lib=Library::new("state/engines/bitnet-build/bin/libllama.so.0.0.1")?;
        let init:Symbol<unsafe extern "C" fn()>=lib.get(b"llama_backend_init")?;init();
        let defaults:Symbol<unsafe extern "C" fn()->Params>=lib.get(b"llama_model_default_params")?;
        let load:Symbol<unsafe extern "C" fn(*const c_char,Params)->*mut c_void>=lib.get(b"llama_model_load_from_file")?;
        let vocab:Symbol<unsafe extern "C" fn(*const c_void)->*const c_void>=lib.get(b"llama_model_get_vocab")?;
        let tokenize:Symbol<unsafe extern "C" fn(*const c_void,*const c_char,i32,*mut i32,i32,bool,bool)->i32>=lib.get(b"llama_tokenize")?;
        let free:Symbol<unsafe extern "C" fn(*mut c_void)>=lib.get(b"llama_model_free")?;
        let mut params=defaults();params.n_gpu_layers=0;params.vocab_only=true;params.kv_overrides=overrides.as_ptr();
        let file=CString::new("offline_bundle/comparative/magicoder-s-ds-6_7b-q4_k_m/magicoder-s-ds-6.7b.Q4_K_M.gguf")?;
        let model=load(file.as_ptr(),params);if model.is_null(){return Err("C++ vocabulary-only load failed".into())}
        let v=vocab(model);let mut result=Vec::new();
        for add_special in [false,true]{let mut ids=vec![0;4096];let n=tokenize(v,prompt.as_ptr().cast(),prompt.len() as i32,ids.as_mut_ptr(),ids.len()as i32,add_special,true);if n<0{free(model);return Err("token buffer insufficient".into())}ids.truncate(n as usize);result.push(ids.into_iter().map(|x|x as u32).collect::<Vec<_>>());}
        free(model);(result.remove(0),result.remove(0))
    };
    let same=native==cpp_without_special&&native==cpp_with_special;
    let divergence=native.iter().zip(&cpp_without_special).position(|(a,b)|a!=b);
    let report=json!({"schema_version":1,"evidence_kind":"vocabulary_only_original_magicoder_tokeniser_parity","tensor_weights_loaded":false,"gpu_used":false,"native_encode_add_special":false,"cpp_parse_special":true,"metadata_compatibility_overrides":{"tokenizer.ggml.model":"gpt2","tokenizer.ggml.pre":"deepseek-coder","tokenizer.ggml.add_bos_token":false},"native_ids":native,"cpp_add_special_false_ids":cpp_without_special,"cpp_add_special_true_ids":cpp_with_special,"first_different_index":divergence,"success":same,"quality_or_learning_claim":false});
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&report)?)?;println!("{}",serde_json::to_string_pretty(&report)?);if !same{return Err("tokeniser mismatch; generation comparison must wait for diagnosis".into())}Ok(())
}
