RESOURCES_FOLDER = "resources"
GITHUB_RELEASE_URL_TRANSFORMERS = "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/"
GITHUB_RELEASE_URL_CNNs = "https://github.com/theMoro/EfficientSED/releases/download/v0.0.1/"

# checkpoints
CHECKPOINT_URLS = {}

# strong
CHECKPOINT_URLS['BEATs_strong_1'] = GITHUB_RELEASE_URL_TRANSFORMERS + "BEATs_strong_1.pt"
CHECKPOINT_URLS['ATST-F_strong_1'] = GITHUB_RELEASE_URL_TRANSFORMERS + "ATST-F_strong_1.pt"
CHECKPOINT_URLS['ASIT_strong_1'] = GITHUB_RELEASE_URL_TRANSFORMERS + "ASIT_strong_1.pt"
CHECKPOINT_URLS['fpasst_strong_1'] = GITHUB_RELEASE_URL_TRANSFORMERS + "fpasst_strong_1.pt"
CHECKPOINT_URLS['M2D_strong_1'] = GITHUB_RELEASE_URL_TRANSFORMERS + "M2D_strong_1.pt"

width_to_seq_model_dim_mapping = {
    '04': '104',
    '06': '160',
    '20': '512',
    '30': '768'
}

for width in ['04', '06', '10', '20', '30']:
    CHECKPOINT_URLS[f'fmn{width}_strong'] = GITHUB_RELEASE_URL_CNNs + f'fmn{width}_strong.pt'

    if width == '10':
        for seq_model in ['gru', 'hybrid', 'tf']:
            for seq_model_dim in ['128', '256', '512', '1024']:
                CHECKPOINT_URLS[f'fmn{width}+{seq_model}-{seq_model_dim}_strong'] = GITHUB_RELEASE_URL_CNNs + f'fmn{width}+{seq_model}-{seq_model_dim}_strong.pt'
    else:
        seq_model_dim_ = width_to_seq_model_dim_mapping[width]
        for seq_model in ['gru', 'hybrid', 'tf']:
            CHECKPOINT_URLS[f'fmn{width}+{seq_model}-{seq_model_dim_}_strong'] = GITHUB_RELEASE_URL_CNNs + f'fmn{width}+{seq_model}-{seq_model_dim_}_strong.pt'


# weak
CHECKPOINT_URLS['BEATs_weak'] = GITHUB_RELEASE_URL_TRANSFORMERS + "BEATs_weak.pt"
CHECKPOINT_URLS['ATST-F_weak'] = GITHUB_RELEASE_URL_TRANSFORMERS + "ATST-F_weak.pt"
CHECKPOINT_URLS['ASIT_weak'] = GITHUB_RELEASE_URL_TRANSFORMERS + "ASIT_weak.pt"
CHECKPOINT_URLS['fpasst_weak'] = GITHUB_RELEASE_URL_TRANSFORMERS + "fpasst_weak.pt"
CHECKPOINT_URLS['M2D_weak'] = GITHUB_RELEASE_URL_TRANSFORMERS + "M2D_weak.pt"

for width in ['04', '06', '10', '20', '30']:
    CHECKPOINT_URLS[f'fmn{width}_weak'] = GITHUB_RELEASE_URL_CNNs + f'fmn{width}_weak.pt'

# ssl
CHECKPOINT_URLS['BEATs_ssl'] = GITHUB_RELEASE_URL_TRANSFORMERS + "BEATs_ssl.pt"
CHECKPOINT_URLS['ATST-F_ssl'] = GITHUB_RELEASE_URL_TRANSFORMERS + "ATST-F_ssl.pt"
CHECKPOINT_URLS['ASIT_ssl'] = GITHUB_RELEASE_URL_TRANSFORMERS + "ASIT_ssl.pt"
CHECKPOINT_URLS['fpasst_ssl'] = GITHUB_RELEASE_URL_TRANSFORMERS + "fpasst_ssl.pt"
CHECKPOINT_URLS['M2D_ssl'] = GITHUB_RELEASE_URL_TRANSFORMERS + "M2D_ssl.pt"

# advanced-kd-weak-strong
CHECKPOINT_URLS['fmn10+tf-256_advanced-kd-weak-strong'] = GITHUB_RELEASE_URL_CNNs + "fmn10+tf-256_advanced-kd-weak-strong.pt"