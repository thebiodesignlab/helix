#!/bin/bash
# DESCRIPTION
#    Install longread_umi as micromamba environment.
#
# IMPLEMENTATION
#    author   Søren Karst (sorenkarst@gmail.com)
#             Ryan Ziels (ziels@mail.ubc.ca)
#    license  GNU General Public License

# Terminal input
BRANCH=${1:-master} # Default to master branch

# Check micromamba installation ----------------------------------------------------
if [[ -z $(which micromamba) ]]; then
  # Install micromamba
  [ -f Mambaforge-Linux-x86_64.sh ] ||\
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh"
  bash ./Mambaforge-Linux-x86_64.sh -b -p $HOME/mambaforge
  export PATH="$HOME/mambaforge/bin:$PATH"
else
  echo ""
  echo "Micromamba found"
  echo "version: $(micromamba --version)"
  echo ""
fi

# Should run not on subshell
eval "$(micromamba shell hook -s bash)"


# Install longread-UMI micromamba env ----------------------------------------------
echo ""
echo "Installing longread_umi micromamba environment.."
echo ""

# Define micromamba env yml
echo "name: longread_umi
channels:
- conda-forge
- bioconda
- defaults
dependencies:
- seqtk=1.3
- parallel=20191122
- racon=1.4.10
- minimap2=2.17
- medaka=0.11.5
- gawk=4.1.3
- cutadapt=2.7
- filtlong=0.2.0
- bwa=0.7.17
- samtools=1.9
- bcftools=1.9
- git
- porechop
" > ./longread_umi.yml

# Install micromamba env
micromamba create -f ./longread_umi.yml
eval "$(micromamba shell hook -s bash)"

micromamba activate longread_umi
if [ $? -eq 0 ]; then
    echo "Environment activated successfully."
else
    echo "Failed to activate environment."
    exit 1
fi

# Install porechop
$CONDA_PREFIX/bin/pip install \
  git+https://github.com/rrwick/Porechop.git   


# Download and install USEARCH
echo "Downloading USEARCH..."
wget -q "https://drive5.com/cgi-bin/upload3.py?license=2024052109533710141&data=05|02||c08234521e624d1b4d2d08dc799d6aa2|4eed7807ebad415aa7a99170947f4eae|0|0|638518964264703727|Unknown|TWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0=|0|||&sdata=Fa9PzLKLAKhw7JsKsvHTEgK0U8d9TK0U9WR0iY1AsBo=&reserved=0" -O usearch
echo "Installing USEARCH..."
chmod +x usearch
mv usearch $CONDA_PREFIX/bin/usearch
echo "USEARCH installation complete."

# Download longread-UMI from git
git clone \
  --branch "$BRANCH" \
  https://github.com/SorenKarst/longread-UMI-pipeline.git \
  $CONDA_PREFIX/longread_umi

# Modify adapters.py
cp \
  $CONDA_PREFIX/longread_umi/scripts/adapters.py \
  $CONDA_PREFIX/lib/python3.6/site-packages/porechop/adapters.py

# Create links to pipeline
find \
  $CONDA_PREFIX/longread_umi/ \
  -name "*.sh" \
  -exec chmod +x {} \;
  
ln -s \
  $CONDA_PREFIX/longread_umi/longread_umi.sh \
  $CONDA_PREFIX/bin/longread_umi


# Check installation
if [[ -z $(which longread_umi) ]]; then
  echo ""
  echo "Can't locate longread_umi"
  echo "longread_umi installation failed..."
  echo ""
else
  echo ""
  echo "longread_umi installation success..."
  echo ""
  echo "Path to micromamba environment: $CONDA_PREFIX"
  echo "Path to pipeline files: $CONDA_PREFIX/longread_umi"
  echo ""
  echo "Initiate micromamba and refresh terminal:"
  echo "micromamba init; source ~/.bashrc"
  echo ""
fi

# Continue with the rest of your script

# Cleanup
if [ -f Mambaforge-Linux-x86_64.sh  ]; then 
  rm -f ./Mambaforge-Linux-x86_64.sh
fi
if [ -f install_conda.sh  ]; then 
  rm -f ./install_conda.sh
fi
if [ -f longread_umi.yml  ]; then 
  rm -f ./longread_umi.yml
fi