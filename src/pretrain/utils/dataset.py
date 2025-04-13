from pretrain.utils import logger
from tqdm import tqdm
from typing import List, Optional
from pretrain.utils.board import WINNER_MAP, UCI_TO_INT_MAP, INT_TO_UCI_MAP

import os
import pyarrow.parquet as pq
import polars as pl
import chess


class DatasetPreprocessor:
    """
    Used for preprocessing and slicing the dataset.
    Assumes the dataset is given in format from https://huggingface.co/datasets/angeluriot/chess_games
    and is parquet file.
    """

    ParquetSchema = pl.Schema(
        {
            "moves_uci": pl.List(pl.Utf8),
            "winner": pl.Utf8,
            "black_elo": pl.Int64,
            "white_elo": pl.Int64,
            "end_type": pl.Utf8,
        }
    )
    CompressedSchema = pl.Schema({
        'actions': pl.List(pl.Int16),
        'winner': pl.Int8,
    })

    def __init__(
            self,
            data_path: str,
            dataset_out_path: str,
            subset_size: float = 0.03,
            elo_strength: float = 1.0,
            num_chunks: Optional[int] = None
    ):
        """
        Used for specified data preparation and initial preprocessing.
        More preprocessing can be in DataModules.
        :param data_path: Path to the parquet dataset file.
        :param dataset_out_path: Path to the output dataset folder.
        :param subset_size: Percentage of dataset to use.
        :param elo_strength: Expected ELO strength of the dataset based on average game ELO.
        Dataset is sorted by ELO and this parameter specifies whether to take last samples (if 1.0),
        which have the highest ELO rating or take the first samples (if 0.0), which have the smallest ELO.
        :param num_chunks: Number of chunks to be saved. If None then no chunks are used. Dataset is saved as a whole.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found under {data_path}")
        os.makedirs(dataset_out_path, exist_ok=True)
        if len(list(filter(lambda key: key != ".DS_Store", os.listdir(dataset_out_path)))):
            logger.warn("Dataset output path already contains some files. They may be overwritten! "
                        f"Files found are : {list(filter(lambda key: key != '.DS_Store', os.listdir(dataset_out_path)))}")

        self.data_path = data_path
        self.dataset_out_path = dataset_out_path
        self.num_chunks = num_chunks
        self.subset_size = subset_size
        self.elo_strength = elo_strength
        self.data_len = pq.ParquetFile(data_path).metadata.num_rows
        self.dataset_df = None
        self.subset_len = None
        self.processing_operation = lambda data: data
        self.progress_bar = None

    def prepare(self):
        """
        Prepares the dataset on a LazyFrame, so it is sorted by average ELO of the game.
        :return:
        """
        dataset_df = pl.scan_parquet(self.data_path, schema=self.ParquetSchema)
        dataset_df = dataset_df.select(["moves_uci", "winner", "end_type", "black_elo", "white_elo"])
        dataset_df = dataset_df.with_columns(
            ((pl.col('black_elo') + pl.col('white_elo')) / 2).alias('avg_elo')
        )
        dataset_df = dataset_df.with_columns(
            pl.when(pl.col('end_type').cast(pl.Utf8) == "draw_agreement")
            .then(pl.lit("draw"))
            .otherwise(pl.col("winner"))
            .alias("winner")
        )
        filter_condition = ~pl.col('avg_elo').is_null() & ~pl.col('moves_uci').is_null() & ~pl.col('winner').is_null()
        dataset_df = dataset_df.filter(filter_condition)
        dataset_df = dataset_df.with_columns(pl.col("avg_elo").cast(pl.Int16))
        dataset_df = dataset_df.select(["avg_elo", "winner", "moves_uci"])
        elo_info = (dataset_df.select("avg_elo").cast(pl.Int16).with_row_index("original_idx")
                    .collect(engine="streaming"))
        sorted_indices = elo_info.sort("avg_elo")["original_idx"].to_numpy()
        size = int(self.data_len * self.subset_size)
        window = self.data_len - size
        start_index = int(window * self.elo_strength)
        self.subset_len = size
        sorted_indices = sorted_indices[start_index:start_index + self.subset_len]
        dataset_df = dataset_df.select(["winner", "moves_uci"]).with_row_index('row_idx').filter(
            pl.col("row_idx").is_in(sorted_indices)
        ).drop("row_idx")

        # Map strings to integers for less memory usage.
        def process_batch(batch_df: pl.DataFrame) -> pl.DataFrame:
            batch_df = batch_df.with_columns(
                pl.col("winner").replace(WINNER_MAP).cast(pl.Int8).alias("winner")
            )
            batch_df = batch_df.with_columns(
                pl.col("moves_uci").map_elements(lambda move_list: [UCI_TO_INT_MAP[move] for move in move_list],
                                                 return_dtype=pl.List(pl.Int64))
                .cast(pl.List(pl.Int16))
                .alias("actions")
            )
            return batch_df.select(["actions", "winner"])
        self.dataset_df = dataset_df.map_batches(process_batch, schema=self.CompressedSchema, streamable=True)
        self.dataset_df = self.dataset_df.collect(engine="streaming")

    def to_fen_dataset(self):
        """
        Converts the dataset into FEN encoded state for each existing board state (as lazy operation).
        Less memory heavy operation, but requires decoding process for neural networks' processing.
        """
        def serialize_moves(moves: List[str]):
            board = chess.Board()
            response = [board.fen()]
            for move in moves:
                board.push_uci(INT_TO_UCI_MAP[move])
                response.append(board.fen())
            return response

        def process(batch_df: pl.DataFrame):
            length = len(batch_df)
            batch_df = batch_df.with_columns(
                pl.col("actions").map_elements(serialize_moves, return_dtype=pl.List(pl.Utf8))
                .alias("states"),
                ).select(["winner", "states", "actions"])
            batch_df = batch_df.with_columns(
                pl.col("actions").list.concat([UCI_TO_INT_MAP["Terminal"]]).cast(pl.List(pl.Int16)).alias("actions")
            )
            batch_df = batch_df.with_row_index("game").explode("states", "actions")
            batch_df = batch_df.with_columns(
                (pl.arange(0, pl.count())).over("game").cast(pl.Int16).alias("move_index")
            )
            batch_df = batch_df.with_columns(
                pl.col("game").cast(pl.UInt64)
            )
            self.progress_bar.update(length)
            return batch_df.select(["states", "winner", "game", "move_index", "actions"])

        self.processing_operation = process

    def preprocess(self):
        """
        Runs the actual preprocessing. May take a while, especially for large `subset_size` and heavy
        memory serialisation. Saves it into the specified directory.
        """
        if self.num_chunks is None:
            chunk_df = self.processing_operation(self.dataset_df)
            chunk_df.write_parquet(os.path.join(self.dataset_out_path, "dataset.parquet"))
        else:
            chunk_size = self.subset_len // self.num_chunks
            for chunk in range(self.num_chunks):
                if chunk == self.num_chunks - 1:
                    chunk_size = self.subset_len - (chunk * chunk_size)

                logger.info(f"Processing chunk {chunk}...")
                self.progress_bar = tqdm(desc="Processing", total=chunk_size, unit="game", smoothing=1.0)
                chunk_df = self.processing_operation(self.dataset_df.slice(chunk * chunk_size, chunk_size))
                self.progress_bar.close()
                chunk_df.write_parquet(os.path.join(self.dataset_out_path, f"dataset_chunk_{chunk}.parquet"))
