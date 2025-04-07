#!/bin/bash
# build_and_run.sh - Build and run script for Chess RL with uv integration
# Użycie:
#   ./build_and_run.sh {build|train|play|dev|watch} [gpu|nogpu]
# Jeśli druga flaga jest "gpu", to uruchomienie będzie z konfiguracją GPU.
# W przeciwnym razie (lub gdy flaga jest "nogpu") zostanie usunięta konfiguracja GPU.

# Ustawienie PATH, aby upewnić się, że katalogi systemowe są dostępne
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Ustawienie flagi GPU (domyślnie CPU-only)
GPU_FLAG="nogpu"
if [ "$2" == "gpu" ]; then
    GPU_FLAG="gpu"
fi

# Build the Docker images
build_images() {
    echo "Building Docker images..."
    docker-compose build
}

# Run the training container
run_training() {
    echo "Starting training container..."
    if [ "$GPU_FLAG" == "gpu" ]; then
        echo "Using GPU configuration."
        docker-compose up train
    else
        echo "Using CPU-only configuration."
        # Remove GPU-specific configuration for CPU-only systems
        # Create a temporary docker-compose file without GPU configuration
        cat docker-compose.yml | \
            grep -v "deploy:" | \
            grep -v "resources:" | \
            grep -v "reservations:" | \
            grep -v "devices:" | \
            grep -v "driver: nvidia" | \
            grep -v "count: 1" | \
            grep -v "capabilities: \[gpu\]" | \
            grep -v "CUDA_VISIBLE_DEVICES=0" | \
            grep -v "gpus:" > docker-compose.cpu.yml

        docker-compose -f docker-compose.cpu.yml up train

        # Clean up the temporary file
        rm docker-compose.cpu.yml
    fi
}

# Run the GUI container
run_gui() {
    echo "Starting GUI container..."
    # Set up X11 forwarding
    xhost +local:docker || true
    docker-compose up play
}

# Run development shell
run_dev() {
    echo "Starting development container..."
    # Set up X11 forwarding
    xhost +local:docker || true
    docker-compose run dev
}

# Run docker-compose with watch mode for development
run_watch() {
    echo "Starting development container with file watching..."
    docker-compose watch
}

# Main execution
case "$1" in
    "build")
        build_images
        ;;
    "train")
        run_training
        ;;
    "play")
        run_gui
        ;;
    "dev")
        run_dev
        ;;
    "watch")
        run_watch
        ;;
    *)
        echo "Usage: $0 {build|train|play|dev|watch} [gpu|nogpu]"
        echo "  build: Build the Docker images"
        echo "  train: Run the training container (use 'gpu' or 'nogpu' as second argument)"
        echo "  play: Run the GUI container"
        echo "  dev: Run a development shell"
        echo "  watch: Run with file watching for development"
        exit 1
        ;;
esac

exit 0