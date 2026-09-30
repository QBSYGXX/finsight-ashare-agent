"""模块：阶段入口；作用：分进程隔离训练、重载、合并和推理；核心业务：否。"""
import argparse
import os
import traceback
from .common import run_path,save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','probe','train','reload-probe','reload-full','merge','base','tuned','compare'])
    parser.add_argument('--run',required=True)
    args=parser.parse_args();dest=run_path(args.run)
    print('STAGE='+args.stage,flush=True)
    try:
        if args.stage=='prepare':
            from .tokenization import prepare
            prepare(dest)
        elif args.stage in ['probe','train']:
            from .train import train
            train(dest,probe=args.stage=='probe')
        elif args.stage.startswith('reload-'):
            from .train import reload_check
            reload_check(dest,probe=args.stage=='reload-probe')
        elif args.stage=='merge':
            from .merge import merge
            merge(dest)
        elif args.stage in ['base','tuned']:
            from .evaluate import evaluate
            evaluate(dest,args.stage)
        else:
            from .evaluate import compare
            compare(dest)
    except Exception as exc:
        save(dest/f"failure-{args.stage}-rank{os.environ.get('RANK','0')}.json",
             {'status':'failed','stage':args.stage,'error':repr(exc),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
