echo "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    echo "Virtual environment exists. Removing existing virtual environment..."
    rm -rf .venv
fi

echo "Creating new virtual environment..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt