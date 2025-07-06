#!/bin/bash

# Crypto Trading AI Agent Setup Script
# This script helps you set up the trading environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "🤖 Crypto Trading AI Agent Setup"
    echo "=================================================="
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}📍 $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Check if Python is installed
check_python() {
    print_step "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
        print_success "Python $PYTHON_VERSION found"
        
        # Check if Python version is 3.8 or higher
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python version is compatible (3.8+)"
        else
            print_error "Python 3.8 or higher is required"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        echo "Please install Python 3.8 or higher from https://python.org"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    print_step "Installing system dependencies..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y build-essential wget curl libta-lib-dev
        elif command -v yum &> /dev/null; then
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y wget curl ta-lib-devel
        else
            print_warning "Please install build tools and TA-Lib manually"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ta-lib
        else
            print_warning "Homebrew not found. Please install TA-Lib manually:"
            echo "1. Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            echo "2. Run: brew install ta-lib"
        fi
    else
        print_warning "Unknown OS. Please install TA-Lib manually"
    fi
}

# Create virtual environment
create_venv() {
    print_step "Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_success "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install Python dependencies
install_python_deps() {
    print_step "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Create environment file
create_env_file() {
    print_step "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        cp .env.template .env
        print_success "Environment file created from template"
        print_warning "Please edit .env file with your API credentials before running the agent"
        echo "Required settings:"
        echo "  - BINANCE_API_KEY: Your Binance API key"
        echo "  - BINANCE_SECRET: Your Binance secret key"
    else
        print_success "Environment file already exists"
    fi
}

# Create necessary directories
create_directories() {
    print_step "Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p data/historical
    mkdir -p data/live
    mkdir -p models
    
    print_success "Directories created"
}

# Test installation
test_installation() {
    print_step "Testing installation..."
    
    # Test imports
    python3 -c "
import pandas as pd
import numpy as np
import yaml
print('✅ Basic imports successful')

try:
    import talib
    print('✅ TA-Lib import successful')
except ImportError:
    print('❌ TA-Lib import failed')
    exit(1)

try:
    import ccxt
    print('✅ CCXT import successful')
except ImportError:
    print('❌ CCXT import failed')
    exit(1)

try:
    import streamlit
    print('✅ Streamlit import successful')
except ImportError:
    print('❌ Streamlit import failed')
    exit(1)

print('✅ All core dependencies are working')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
    else
        print_error "Installation test failed"
        exit 1
    fi
}

# Run configuration check
check_config() {
    print_step "Checking configuration..."
    
    if [ -f "config.yaml" ]; then
        python3 -c "
import yaml
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    required_sections = ['exchange', 'trading', 'strategy']
    for section in required_sections:
        if section not in config:
            print(f'❌ Missing required section: {section}')
            exit(1)
    
    print('✅ Configuration file is valid')
except Exception as e:
    print(f'❌ Configuration error: {e}')
    exit(1)
"
        print_success "Configuration check passed"
    else
        print_error "config.yaml not found"
        exit 1
    fi
}

# Show next steps
show_next_steps() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "🎉 Setup Complete!"
    echo "=================================================="
    echo -e "${NC}"
    
    echo "Next steps:"
    echo ""
    echo "1. 📝 Edit your environment file:"
    echo "   nano .env"
    echo ""
    echo "2. 🔑 Add your Binance API credentials:"
    echo "   - BINANCE_API_KEY=your_api_key"
    echo "   - BINANCE_SECRET=your_secret_key"
    echo ""
    echo "3. 🧪 Test with paper trading:"
    echo "   python main.py --mode paper --strategy rule_based"
    echo ""
    echo "4. 📊 Launch the dashboard:"
    echo "   streamlit run dashboard/app.py"
    echo ""
    echo "5. 🔙 Run backtests:"
    echo "   python main.py --mode backtest --strategy rule_based"
    echo ""
    echo "6. 📚 Read the documentation:"
    echo "   cat README.md"
    echo ""
    
    print_warning "Important Security Notes:"
    echo "- Never share your API keys"
    echo "- Start with paper trading mode"
    echo "- Use small amounts for initial live testing"
    echo "- Enable IP restrictions on your Binance account"
    echo ""
    
    print_success "Happy trading! 🚀"
}

# Docker setup
setup_docker() {
    print_step "Setting up Docker environment..."
    
    if command -v docker &> /dev/null; then
        print_success "Docker found"
        
        # Build the image
        docker build -t crypto-trading-ai .
        
        print_success "Docker image built successfully"
        
        echo "To run with Docker:"
        echo "  docker-compose up -d"
        echo ""
        echo "To view logs:"
        echo "  docker-compose logs -f trading-agent"
        
    else
        print_warning "Docker not found. Install Docker to use containerized deployment"
    fi
}

# Main execution
main() {
    print_header
    
    # Parse command line arguments
    DOCKER_SETUP=false
    SKIP_DEPS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --docker)
                DOCKER_SETUP=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --docker      Setup Docker environment"
                echo "  --skip-deps   Skip system dependency installation"
                echo "  --help        Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check Python
    check_python
    
    # Install system dependencies (unless skipped)
    if [ "$SKIP_DEPS" = false ]; then
        install_system_deps
    fi
    
    # Create virtual environment
    create_venv
    
    # Install Python dependencies
    install_python_deps
    
    # Create environment file
    create_env_file
    
    # Create directories
    create_directories
    
    # Test installation
    test_installation
    
    # Check configuration
    check_config
    
    # Setup Docker if requested
    if [ "$DOCKER_SETUP" = true ]; then
        setup_docker
    fi
    
    # Show next steps
    show_next_steps
}

# Check if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
