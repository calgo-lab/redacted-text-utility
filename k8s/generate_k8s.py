from pathlib import Path

import jinja2

if __name__ == '__main__':
    image = "registry.datexis.com/troehr/text_redact"
    seeds = [1312]
    redaction_types = ['text_redacted_with_semantic_label_mask', 'text_redacted_with_random_mask', 'text_redacted_with_generic_mask', "none"] 
    model_name = "microsoft/BiomedNLP-biomedbert-base-uncased-abstract-fulltext"
    hidden_dim = 768
    dataset_path = "/pvc/data/processed"
    dataset_stem = "-00000-of-00001_ne_redacted.parquet"
    logs_root = '/pvc/logs/biomedbert'
    batch_size = 8
    warmup_steps = 100
    mode = "test"
    save_scores = False if mode == "fit" else True 
    ckpt_path = None
    test_out = ""
    lr = 2e-5
    hidden_dim = 768
    k8s_out = f"/Users/toroe/Workspace/redacted-text-utility/k8s/{mode}"
    with open(Path(__file__).parent / f'template_{mode}.yaml') as f:
        template = jinja2.Template(f.read())
    for redaction_type in redaction_types:
        for seed in seeds:
            seed_model_redaction_name = f"{str(seed)}-{logs_root.split('/')[-1]}-{redaction_type.replace('_', '-')}"
            short_name = f"{str(seed)}-{redaction_type.replace('_', '-').replace('text-redacted-with-', '')}"
            out_path = f"{k8s_out}/{seed_model_redaction_name}" 
            if mode == "test":
                ckpt_path = f"{logs_root}/{seed_model_redaction_name}/checkpoint.ckpt"
                for redaction_type in redaction_types:
                    test_out = f"/pvc/experiments/{seed_model_redaction_name}-on-{redaction_type}"
                    testout_path = out_path + f"-on-{redaction_type}"
                    with open(f'{testout_path}_{mode}.yaml', "w") as fh:
                        
                        fh.write(template.render({
                            
                            'mode': mode,
                            'image': image,
                            'experiment': f"{short_name}-{mode}-{redaction_type.replace('_', '-').replace('text-redacted-with-', '')}",
                            'root_dir': f"{logs_root}/{seed_model_redaction_name}-on-{redaction_type}",
                            'batch_size': batch_size,
                            'model_name': model_name,
                            'hidden_dim': hidden_dim,
                            'warmup_steps': warmup_steps,
                            'lr': lr,
                            'seed': seed,
                            'dataset_path': dataset_path,
                            'dataset_stem': dataset_stem,
                            'redaction_type': redaction_type if redaction_type != "none" else None,
                            'save_scores': save_scores,
                            'test_out': test_out,
                            'ckpt':ckpt_path
                        }))
            else:
                with open(f'{out_path}_{mode}.yaml', "w") as fh:
                    
                    fh.write(template.render({
                        
                        'mode': mode,
                        'image': image,
                        'experiment': f"{seed_model_redaction_name}-{mode}",
                        'root_dir': f"{logs_root}/{seed_model_redaction_name}",
                        'batch_size': batch_size,
                        'model_name': model_name,
                        'hidden_dim': hidden_dim,
                        'warmup_steps': warmup_steps,
                        'lr': lr,
                        'seed': seed,
                        'dataset_path': dataset_path,
                        'dataset_stem': dataset_stem,
                        'redaction_type': redaction_type if redaction_type != "none" else None,
                    }))
