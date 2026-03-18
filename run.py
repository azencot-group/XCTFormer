import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description='XCTFormer')

    # Basic config
    parser.add_argument('--task_name', type=str, required=True,
                        help='task name: long_term_forecast, imputation, anomaly_detection')
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True, default='XCTFormer',
                        help='model name: XCTFormer')

    # Data
    parser.add_argument('--data', type=str, required=True, default='ETTm1')
    parser.add_argument('--root_path', type=str, default='./datasets/ETT-small/')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv')
    parser.add_argument('--features', type=str, default='M',
                        help='M: multivariate, S: univariate, MS: multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)

    # Imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25)

    # Anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=1.0)

    # Model
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)

    # XCTFormer-specific
    parser.add_argument('--include_decop', action='store_true', default=False,
                        help='Use decomposed compressed attention (DeCop) mode')
    parser.add_argument('--k', type=int, default=0,
                        help='Compressed dimension for DeCop mode')

    # Training
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lradj', type=str, default='TST')
    parser.add_argument('--pct_start', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--handle_end_epoch_rate', type=int, default=1)

    # Regularization
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--fc_dropout', type=float, default=0.05)
    parser.add_argument('--head_dropout', type=float, default=0.0)

    # Other
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1')
    parser.add_argument('--inverse', action='store_true', default=False)

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('Args in experiment:')
    print(args)

    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'imputation':
        from exp.exp_imputation import Exp_Imputation
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        Exp = Exp_Anomaly_Detection
    else:
        raise ValueError(f'Unknown task: {args.task_name}')

    if args.is_training:
        setting = f'{args.model_id}_{args.task_name}_{args.data}'
        exp = Exp(args)
        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)
        torch.cuda.empty_cache()
    else:
        setting = f'{args.model_id}_{args.task_name}_{args.data}'
        exp = Exp(args)
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
