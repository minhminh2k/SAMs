from typing import Any, Dict, Optional, Tuple

import albumentations as A
import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from src.data.coco.components.coco_dataset import COCODataset, collate_fn
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class COCODataModule(LightningDataModule):
    """`LightningDataModule` for the COCO dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_train_test_dir: str = "data/train",
        train_test_annotation: str = "data/train/annotations",
        data_val_dir: str = "data/val",
        val_annotation: str = "data/val/annotations",
        train_test_subset: int = 20000,
        val_subset: int = 2500,
        train_test_split: Tuple[int, int] = (0.9, 0.1),
        transform_train: Optional[A.Compose] = None,
        transform_val: Optional[A.Compose] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """Initialize a `COCODataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.batch_size_per_device = batch_size
    
    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train_test_dataset = COCODataset(
                data_dir=self.hparams.data_train_test_dir,
                annotation_file=self.hparams.train_test_annotation
            )
            val_dataset = COCODataset(
                data_dir=self.hparams.data_val_dir,
                annotation_file=self.hparams.val_annotation
            )
            
            train_test_indices = torch.randperm(len(train_test_dataset))[:self.hparams.train_test_subset]
            val_indices = torch.randperm(len(val_dataset))[:self.hparams.val_subset]
            
            logging.info(f"Train Test Subset: {len(train_test_indices)}")
            logging.info(f"Val Subset: {len(val_indices)}")
            
            train_test_dataset = Subset(train_test_dataset, train_test_indices)
            val_dataset = Subset(val_dataset, val_indices)
            
            train_test_len = len(train_test_dataset)
            train_len = int(train_test_len * self.hparams.train_test_split[0])
            test_len = train_test_len - train_len


            self.data_train, self.data_test = random_split(
                dataset=train_test_dataset,
                lengths=[train_len, test_len],
                generator=torch.Generator().manual_seed(42),
            )
            self.data_val = val_dataset

            logging.info("Using random_split for Train and Test Dataset")
            logging.info(f"Train Dataloader: {len(self.data_train)}")
            logging.info(f"Val Dataloader: {len(self.data_val)}")
            logging.info(f"Test Dataloader: {len(self.data_test)}")

            
            

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )
        
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    import hydra
    import rootutils
    from omegaconf import DictConfig, OmegaConf

    path = rootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs")
    output_path = path / "outputs"
    print(f"config_path: {config_path}")
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    @hydra.main(version_base="1.3", config_path=config_path, config_name="train.yaml")
    def main(cfg: DictConfig):
        print(OmegaConf.to_yaml(cfg.data, resolve=True))
        
        coco = hydra.utils.instantiate(cfg.data)
        loader = coco.test_dataloader()
        images, bboxes, masks = next(iter(loader))
        print(images.shape)
        print(bboxes.shape)
        print(masks.shape)
        
    main()