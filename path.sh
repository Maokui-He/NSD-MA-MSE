#[ -d /home/backman/tools/kaldi ] && export KALDI_ROOT=/home/backman/tools/kaldi
#[ -d /home/backman//kaldi ] && export KALDI_ROOT=/home/backman//kaldi
export KALDI_ROOT=/disk1/yyy/kaldi
#export PATH="/disk1/mkhe/anaconda3/bin:$PATH"
#[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
#[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
[ -f $KALDI_ROOT/tools/config/common_path.sh ] && . $KALDI_ROOT/tools/config/common_path.sh

export LC_ALL=C


