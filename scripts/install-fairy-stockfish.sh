#!/usr/bin/env bash
set -euo pipefail

readonly FAIRY_REPOSITORY="https://github.com/fairy-stockfish/Fairy-Stockfish.git"
readonly FAIRY_COMMIT="f3e6969d11d1bec17eba26e7ae0e629ad4af71dd"
readonly DESTINATION="${1:-vendor/stockfish/fairy-stockfish-14}"
readonly BUILD_JOBS="${FAIRY_STOCKFISH_BUILD_JOBS:-8}"

build_dir=$(mktemp -d)
staged_destination=""
cleanup() {
    rm -rf -- "$build_dir"
    if [[ -n "$staged_destination" ]]; then
        rm -f -- "$staged_destination"
    fi
}
trap cleanup EXIT

git clone --quiet --filter=blob:none --no-checkout "$FAIRY_REPOSITORY" "$build_dir/source"
git -C "$build_dir/source" checkout --quiet --detach "$FAIRY_COMMIT"

actual_commit=$(git -C "$build_dir/source" rev-parse HEAD)
if [[ "$actual_commit" != "$FAIRY_COMMIT" ]]; then
    echo "Fairy-Stockfish source commit mismatch: $actual_commit" >&2
    exit 1
fi

make -C "$build_dir/source/src" -j"$BUILD_JOBS" profile-build ARCH=x86-64-bmi2

readonly BUILT_BINARY="$build_dir/source/src/stockfish"
uci_output=$(printf 'uci\nquit\n' | "$BUILT_BINARY")
if ! grep -Fq 'id name Fairy-Stockfish 14' <<<"$uci_output"; then
    echo "Built binary does not identify as Fairy-Stockfish 14" >&2
    exit 1
fi
if ! grep -Fq 'option name UCI_Elo type spin default 1350 min 500 max 2850' <<<"$uci_output"; then
    echo "Built binary does not expose the pinned UCI_Elo 500..2850 contract" >&2
    exit 1
fi

destination_directory=$(dirname -- "$DESTINATION")
destination_name=$(basename -- "$DESTINATION")
install -d "$destination_directory"
staged_destination="$destination_directory/.${destination_name}.tmp-$$"
install -m755 "$BUILT_BINARY" "$staged_destination"
mv -f -- "$staged_destination" "$DESTINATION"
staged_destination=""

sha256sum "$DESTINATION"
