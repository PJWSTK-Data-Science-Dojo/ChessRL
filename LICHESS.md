# Run Luna on Lichess

Luna exposes a standard UCI engine and uses the maintained
[`lichess-bot`](https://github.com/lichess-bot-devs/lichess-bot) bridge for network play. The generated configuration is deliberately conservative: one game at a time, standard chess, no pondering, no takebacks, and challenge time controls from blitz through classical.

## Scope

This integration is for games delivered through the official Lichess Bot API and direct challenges. It does not put a bot into the normal lobby pools, and it does not configure ordinary Arena or Swiss tournament enrollment. In this project, “arena-ready” means that the process can stay online and accept challenge games reliably.

Bot conversion is irreversible, and the account must never have played a game before conversion. Lichess documents those constraints in the
[`botAccountUpgrade` API](https://lichess.org/api#operation/botAccountUpgrade).

## 1. Prepare both projects

Run these commands from the ChessRL repository root:

```bash
uv sync --extra perf
test -x .venv/bin/luna-uci
test -f temp/latest.pth.tar

git clone https://github.com/lichess-bot-devs/lichess-bot.git ../lichess-bot
python3 -m venv ../lichess-bot/venv
../lichess-bot/venv/bin/python -m pip install -r ../lichess-bot/requirements.txt
```

The UCI executable and checkpoint are recorded as absolute paths, so the two repositories can live anywhere. Luna loads `temp/latest.pth.tar`; every new game therefore starts with the latest atomically published model.

Smoke-test the engine before connecting it to an account:

```bash
printf 'uci\nisready\nquit\n' | .venv/bin/luna-uci \
  --checkpoint "$(pwd)/temp/latest.pth.tar" \
  --device cuda \
  --mcts-sims 32
```

The output must include `uciok` and `readyok`.

## 2. Create the bot identity and token

Create a fresh Lichess account that has played no games. While signed into that account, create an OAuth token with only the `bot:play` (“Play games with the bot API”) scope.

Read the token without echoing it or placing it in shell history:

```bash
read -rsp 'Lichess bot token: ' LICHESS_TOKEN && printf '\n'
export LICHESS_TOKEN
```

Never paste the token into a command, commit it, attach it to an issue, or include it in logs.

## 3. Generate `config.yml`

The generator reads the checked-out upstream `config.yml.default` with PyYAML, retains unknown upstream fields, and writes an owner-only file atomically:

```bash
uv run luna-lichess-config \
  --lichess-bot-dir ../lichess-bot \
  --device cuda \
  --mcts-sims 100 \
  --minimum-sims 8 \
  --estimated-sim-ms 4 \
  --compile-inference

unset LICHESS_TOKEN
stat -c '%a %n' ../lichess-bot/config.yml
```

The mode reported by `stat` must be `600`. The generator never prints the token and refuses to overwrite an existing config unless `--force` is supplied.

Useful options:

- `--checkpoint /absolute/path/to/checkpoint.pth.tar` selects another format-v2 checkpoint.
- `--device cuda|mps|cpu` chooses the inference backend.
- `--cuda-device N` pins the process to one CUDA device.
- `--mcts-sims N` sets the maximum search budget.
- `--minimum-sims N` keeps at least a small search under clock pressure.
- `--estimated-sim-ms FLOAT` converts available move time into a simulation budget.
- `--compile-inference` enables compiled inference after its initial warm-up.

The generated challenge policy accepts standard chess in rated or casual mode and permits only one active game and one game per challenger. Bullet is disabled because safe neural-search latency depends on the host and cannot be enforced reliably for every human challenge. Edit the generated YAML only if you understand the current
[`lichess-bot` configuration schema](https://github.com/lichess-bot-devs/lichess-bot/blob/master/config.yml.default).

## 4. Convert and run the account

The first invocation upgrades the unused account permanently:

```bash
cd ../lichess-bot
./venv/bin/python lichess-bot.py -u
```

For every later start, omit `-u`:

```bash
cd ../lichess-bot
./venv/bin/python lichess-bot.py
```

Open the bot profile on Lichess and send it a direct challenge. Both human and bot challenges are enabled by default.

## Time-control tuning

Start with incremented blitz while measuring real search time. Luna uses the smaller of the configured maximum and the amount affordable from the UCI clock. `--estimated-sim-ms` should be a conservative measured value, including Python and tree-search overhead.

If the bot moves too slowly, increase `--estimated-sim-ms` or reduce `--mcts-sims`. If it consistently has unused time, lower the estimate gradually. Compiled inference can make the first search slower while compilation completes, so warm the engine before accepting short challenges.

The generated config disables pondering and reserves 250 ms of bridge overhead. Those settings favor clock safety over squeezing out one more search batch.

## Operations and security

- Revoke the OAuth token immediately if `config.yml` is exposed. Generate a new token, rerun the configurator with `--force`, then unset the environment variable.
- Keep `config.yml` outside this repository. It contains a live credential even though its permissions are restricted.
- Run only one bridge process for the account. Duplicate processes can race over the same event stream.
- Use `Ctrl+C` for a clean shutdown. The upstream bridge owns network retries and rate-limit handling.
- Retain `latest.pth.tar` while the bot is online. A newly spawned engine cannot start without it.

For protocol diagnostics, run the bridge with `-v`; check that Luna answers `uci`, `isready`, `position`, clocked `go`, and restricted `go searchmoves` commands. Do not share logs until you have checked them for account or filesystem information.

## Sources

- [Official Lichess Bot API](https://lichess.org/api#tag/Bot)
- [Official bot-account upgrade contract](https://lichess.org/api#operation/botAccountUpgrade)
- [Maintained lichess-bot bridge](https://github.com/lichess-bot-devs/lichess-bot)
- [Current upstream configuration template](https://github.com/lichess-bot-devs/lichess-bot/blob/master/config.yml.default)
