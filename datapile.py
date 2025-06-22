import lightning as L
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import datasets
from datasets.distributed import split_dataset_by_node
import h5py

def swap(array, i, j):
    array[[i, j]] = array[[j, i]]
    return array

def swap_tensor(tensor, i, j):
    index = torch.arange(tensor.size(0))
    index[i], index[j] = index[j], index[i]
    return tensor[index]

class LoadRawDataset(Dataset):
    def __init__(self, csv_file, augment=False):
        self.file_paths = self.load_csv(csv_file)
        self.augment = augment

    def load_csv(self, csv_file):
        csv_data = pd.read_csv(csv_file)
        
        if 'file_path' not in csv_data.columns:
            raise ValueError(f"'file_path' column not found in {csv_file}")
        
        file_paths = csv_data['file_path'].tolist()   
        return file_paths 

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        data = np.einsum('ijkl->lijk', data)
        LPT = data[6:9,:,:,:]  
        Nbody = data[0:3,:,:,:]  
        if self.augment:
            if np.random.rand() < .5:
                LPT = LPT[:, ::-1, ...]
                LPT[0] = -LPT[0]
                Nbody = Nbody[:, ::-1, ...]
                Nbody[0] = -Nbody[0]
            if np.random.rand() < .5:
                LPT = LPT[:, :, ::-1, ...]
                LPT[1] = -LPT[1]
                Nbody = Nbody[:, :, ::-1, ...]
                Nbody[1] = -Nbody[1]
            if np.random.rand() < .5:
                LPT = LPT[:, :, :, ::-1]
                LPT[2] = -LPT[2]
                Nbody = Nbody[:, :, :, ::-1]
                Nbody[2] = -Nbody[2]
            prand = np.random.rand()
            if prand < 1./6:
                LPT = np.transpose(LPT, axes=(0, 2, 3, 1))
                LPT = swap(LPT, 0, 2)
                LPT = swap(LPT, 0, 1)
                Nbody = np.transpose(Nbody, axes=(0, 2, 3, 1))
                Nbody = swap(Nbody, 0, 2)
                Nbody = swap(Nbody, 0, 1)
            elif prand < 2./6:
                LPT = np.transpose(LPT, axes=(0, 2, 1, 3))
                LPT = swap(LPT, 0, 1)
                Nbody = np.transpose(Nbody, axes=(0, 2, 1, 3))
                Nbody = swap(Nbody, 0, 1)
            elif prand < 3./6:
                LPT = np.transpose(LPT, axes=(0, 1, 3, 2))
                LPT = swap(LPT, 1, 2)
                Nbody = np.transpose(Nbody, axes=(0, 1, 3, 2))
                Nbody = swap(Nbody, 1, 2)
            elif prand < 4./6:
                LPT = np.transpose(LPT, axes=(0, 3, 1, 2))
                LPT = swap(LPT, 1, 2)
                LPT = swap(LPT, 0, 1)
                Nbody = np.transpose(Nbody, axes=(0, 3, 1, 2))
                Nbody = swap(Nbody, 1, 2)
                Nbody = swap(Nbody, 0, 1)
            elif prand < 5./6:
                LPT = np.transpose(LPT, axes=(0, 3, 2, 1))
                LPT = swap(LPT, 0, 2)
                Nbody = np.transpose(Nbody, axes=(0, 3, 2, 1))
                Nbody = swap(Nbody, 0, 2)

        # Convert to PyTorch tensors
        LPT = torch.from_numpy(LPT.copy())
        Nbody = torch.from_numpy(Nbody.copy())

        return LPT, Nbody

class FastPMPile(L.LightningDataModule): 
    def __init__(
        self,
        train_csv_file: str = None,
        val_csv_file: str = None,
        test_csv_file: str = None,
        batch_size: int = 512,
        num_workers: int = 10,
        augment: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['kwargs'])

    def setup(self, stage):
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.batch_size = self.hparams.batch_size // dist.get_world_size()
        else:
            self.batch_size = self.hparams.batch_size

    def train_dataloader(self):
        train_dataset = LoadRawDataset(csv_file=self.hparams.train_csv_file, augment=self.hparams.augment)
        pin_memory = torch.cuda.is_available()
        return DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            #persistent_workers=True,
            drop_last=True,
            #pin_memory=pin_memory,
        )

    def val_dataloader(self):
        val_dataset = LoadRawDataset(csv_file=self.hparams.train_csv_file, augment=False)
        pin_memory = torch.cuda.is_available()
        return DataLoader(
            val_dataset,
            shuffle=False, 
            batch_size=self.batch_size,
            drop_last=False,  
            #pin_memory=pin_memory,
            num_workers=self.hparams.num_workers,
            #persistent_workers=True,
        )
    
    def test_dataloader(self):
        test_dataset = LoadRawDataset(csv_file=self.hparams.train_csv_file, augment=False)
        pin_memory = torch.cuda.is_available()
        return DataLoader(
            test_dataset,
            shuffle=False,  
            batch_size=self.batch_size,
            drop_last=False,  
            #pin_memory=pin_memory,
            num_workers=self.hparams.num_workers,
            #persistent_workers=True,
        )

class AugmentedDataset(Dataset):
    def __init__(self, dataset, augment=False, velocity=False, density=False, init_density=False, style=None, redshift=None):
        self.dataset = dataset
        self.augment = augment
        self.velocity = velocity
        self.density = density
        self.init_density = init_density
        self.style = style
        self.redshift = redshift

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dset = self.dataset[idx]

        if self.density:
            if self.init_density:
                initial_density = dset['init_density']
                density = dset['density']
                initial_density = np.einsum('ijkl->lijk', initial_density)
                density = np.einsum('ijkl->lijk', density)
                LPT = initial_density[0].unsqueeze(0).numpy().astype(np.float32)
                Nbody = density[1].unsqueeze(0).numpy().astype(np.float32)
            else:
                density = dset['density']
                LPT = density[3].unsqueeze(0).numpy().astype(np.float32)
                Nbody = density[1].unsqueeze(0).numpy().astype(np.float32)                
            if self.augment:
                # Apply axis flips (no sign changes for density)
                if np.random.rand() < .5:
                    LPT = LPT[:, ::-1, ...]  # Flip x-axis
                    Nbody = Nbody[:, ::-1, ...]
                if np.random.rand() < .5:
                    LPT = LPT[:, :, ::-1, ...]  # Flip y-axis
                    Nbody = Nbody[:, :, ::-1, ...]
                if np.random.rand() < .5:
                    LPT = LPT[:, :, :, ::-1]  # Flip z-axis
                    Nbody = Nbody[:, :, :, ::-1]
                
                # Apply axis permutations (no component swaps needed)
                prand = np.random.rand()
                if prand < 1./6:
                    LPT = np.transpose(LPT, axes=(0, 2, 3, 1))  # Permute axes
                    Nbody = np.transpose(Nbody, axes=(0, 2, 3, 1))
                elif prand < 2./6:
                    LPT = np.transpose(LPT, axes=(0, 2, 1, 3))
                    Nbody = np.transpose(Nbody, axes=(0, 2, 1, 3))
                elif prand < 3./6:
                    LPT = np.transpose(LPT, axes=(0, 1, 3, 2))
                    Nbody = np.transpose(Nbody, axes=(0, 1, 3, 2))
                elif prand < 4./6:
                    LPT = np.transpose(LPT, axes=(0, 3, 1, 2))
                    Nbody = np.transpose(Nbody, axes=(0, 3, 1, 2))
                elif prand < 5./6:
                    LPT = np.transpose(LPT, axes=(0, 3, 2, 1))
                    Nbody = np.transpose(Nbody, axes=(0, 3, 2, 1))
        elif self.velocity:
            velocity = dset['velocity']
            velocity = np.einsum('ijkl->lijk', velocity)
            LPT = velocity[6:9].astype(np.float32)
            Nbody = velocity[0:3].astype(np.float32)
            if self.augment:
                if np.random.rand() < .5:
                    LPT = LPT[:, ::-1, ...]
                    LPT[0] = -LPT[0]
                    Nbody = Nbody[:, ::-1, ...]
                    Nbody[0] = -Nbody[0]
                if np.random.rand() < .5:
                    LPT = LPT[:, :, ::-1, ...]
                    LPT[1] = -LPT[1]
                    Nbody = Nbody[:, :, ::-1, ...]
                    Nbody[1] = -Nbody[1]
                if np.random.rand() < .5:
                    LPT = LPT[:, :, :, ::-1]
                    LPT[2] = -LPT[2]
                    Nbody = Nbody[:, :, :, ::-1]
                    Nbody[2] = -Nbody[2]
                prand = np.random.rand()
                if prand < 1./6:
                    LPT = np.transpose(LPT, axes=(0, 2, 3, 1))
                    LPT = swap(LPT, 0, 2)
                    LPT = swap(LPT, 0, 1)
                    Nbody = np.transpose(Nbody, axes=(0, 2, 3, 1))
                    Nbody = swap(Nbody, 0, 2)
                    Nbody = swap(Nbody, 0, 1)
                elif prand < 2./6:
                    LPT = np.transpose(LPT, axes=(0, 2, 1, 3))
                    LPT = swap(LPT, 0, 1)
                    Nbody = np.transpose(Nbody, axes=(0, 2, 1, 3))
                    Nbody = swap(Nbody, 0, 1)
                elif prand < 3./6:
                    LPT = np.transpose(LPT, axes=(0, 1, 3, 2))
                    LPT = swap(LPT, 1, 2)
                    Nbody = np.transpose(Nbody, axes=(0, 1, 3, 2))
                    Nbody = swap(Nbody, 1, 2)
                elif prand < 4./6:
                    LPT = np.transpose(LPT, axes=(0, 3, 1, 2))
                    LPT = swap(LPT, 1, 2)
                    LPT = swap(LPT, 0, 1)
                    Nbody = np.transpose(Nbody, axes=(0, 3, 1, 2))
                    Nbody = swap(Nbody, 1, 2)
                    Nbody = swap(Nbody, 0, 1)
                elif prand < 5./6:
                    LPT = np.transpose(LPT, axes=(0, 3, 2, 1))
                    LPT = swap(LPT, 0, 2)
                    Nbody = np.transpose(Nbody, axes=(0, 3, 2, 1))
                    Nbody = swap(Nbody, 0, 2)            
        else:
            displacement = dset['displacement']
            displacement = np.einsum('ijkl->lijk', displacement)
            LPT = displacement[6:9].astype(np.float32)
            Nbody = displacement[0:3].astype(np.float32)
            if self.augment:
                if np.random.rand() < .5:
                    LPT = LPT[:, ::-1, ...]
                    LPT[0] = -LPT[0]
                    Nbody = Nbody[:, ::-1, ...]
                    Nbody[0] = -Nbody[0]
                if np.random.rand() < .5:
                    LPT = LPT[:, :, ::-1, ...]
                    LPT[1] = -LPT[1]
                    Nbody = Nbody[:, :, ::-1, ...]
                    Nbody[1] = -Nbody[1]
                if np.random.rand() < .5:
                    LPT = LPT[:, :, :, ::-1]
                    LPT[2] = -LPT[2]
                    Nbody = Nbody[:, :, :, ::-1]
                    Nbody[2] = -Nbody[2]
                prand = np.random.rand()
                if prand < 1./6:
                    LPT = np.transpose(LPT, axes=(0, 2, 3, 1))
                    LPT = swap(LPT, 0, 2)
                    LPT = swap(LPT, 0, 1)
                    Nbody = np.transpose(Nbody, axes=(0, 2, 3, 1))
                    Nbody = swap(Nbody, 0, 2)
                    Nbody = swap(Nbody, 0, 1)
                elif prand < 2./6:
                    LPT = np.transpose(LPT, axes=(0, 2, 1, 3))
                    LPT = swap(LPT, 0, 1)
                    Nbody = np.transpose(Nbody, axes=(0, 2, 1, 3))
                    Nbody = swap(Nbody, 0, 1)
                elif prand < 3./6:
                    LPT = np.transpose(LPT, axes=(0, 1, 3, 2))
                    LPT = swap(LPT, 1, 2)
                    Nbody = np.transpose(Nbody, axes=(0, 1, 3, 2))
                    Nbody = swap(Nbody, 1, 2)
                elif prand < 4./6:
                    LPT = np.transpose(LPT, axes=(0, 3, 1, 2))
                    LPT = swap(LPT, 1, 2)
                    LPT = swap(LPT, 0, 1)
                    Nbody = np.transpose(Nbody, axes=(0, 3, 1, 2))
                    Nbody = swap(Nbody, 1, 2)
                    Nbody = swap(Nbody, 0, 1)
                elif prand < 5./6:
                    LPT = np.transpose(LPT, axes=(0, 3, 2, 1))
                    LPT = swap(LPT, 0, 2)
                    Nbody = np.transpose(Nbody, axes=(0, 3, 2, 1))
                    Nbody = swap(Nbody, 0, 2)
        if self.style is not None:
            style_list=[]
            for style in self.style:
                style_data = dset[style]
                style_list.append(style_data)
            if self.redshift is not None:
                style_list.append(self.redshift)
            style_tensor = torch.tensor(style_list, dtype=torch.float32)
            return torch.from_numpy(LPT.copy()), torch.from_numpy(Nbody.copy()), style_tensor
        return torch.from_numpy(LPT.copy()), torch.from_numpy(Nbody.copy())

class HuggingfaceLoader(L.LightningDataModule):  # use to load huggingface dataset
    def __init__(
        self,
        dataset_path: str,
        test_size: float = 0.1,
        shuffle: bool = True,
        batch_size: int = 512,
        num_workers: int = 10,
        augment: bool = True,
        velocity: bool = False,
        density: bool = False,
        init_density: bool = False,
        style: list = None,
        redshift: float = None,
        **kwargs,
    ) -> None:
        """The `HuggingfaceLoader` class defines a LightningDataModule 

        Args:
            dataset_path (`str`):
                The path of the Huggingface dataset folder.
            test_size (`float`, *optional*, defaults to 0.2):
                Used to spit the dataset
            shuffle (`bool`, *optional*, defaults to False):
                Shuffle the dataset.
            batch_size (`int`, *optional*, defaults to 512):
                The global batch size, it will be divided by the number of GPUs.
            num_workers (`int`, *optional*, defaults to 10):
                The number of workers for the DataLoader.
            augment (`bool`, *optional*, defaults to True):
                Augment the dataset.
            velocity (`bool`, *optional*, defaults to True):
                Use density field.   
            density (`bool`, *optional*, defaults to True):
                Use density field.       
            init_density (`bool`, *optional*, defaults to True):
                Use init density field as one of the input.    
            style (`list`, *optional*, defaults to None):
                List of styles to be used for style injection.   
            redshift (`float`, *optional*, defaults to None): #depleted
                Redshift value to be used for style injection.
        """
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage):
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.batch_size = self.hparams.batch_size // dist.get_world_size()
        else:
            self.batch_size = self.hparams.batch_size

        self.dset = datasets.load_from_disk(self.hparams.dataset_path)

        if self.is_distributed:
            self.dset = split_dataset_by_node(
                self.dset, rank=dist.get_rank(), world_size=dist.get_world_size()
            )
        if stage == "validation":
            self.dset = self.dset.train_test_split(
                test_size=0.001,
                shuffle=False,
            )
        else:
            self.dset = self.dset.train_test_split(
                test_size=self.hparams.test_size,
                shuffle=False,
            )

        #self.dset = self.dset.with_format("torch")

        self.train_dataset = AugmentedDataset(self.dset["train"], augment=self.hparams.augment, velocity=self.hparams.velocity, density=self.hparams.density, init_density=self.hparams.init_density, style=self.hparams.style, redshift=self.hparams.redshift)
        self.val_dataset = AugmentedDataset(self.dset["test"], augment=False, velocity=self.hparams.velocity, density=self.hparams.density, init_density=self.hparams.init_density, style=self.hparams.style, redshift=self.hparams.redshift)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
            #num_workers=0,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            #num_workers=0,
        )
    

class HDF5AugmentedDataset(Dataset):
    def __init__(self, csv_file, augment=False, velocity=False, density=False, init_density=False, style=None, redshift=None):
        self.data_info = pd.read_csv(csv_file)
        self.augment = augment
        self.velocity = velocity
        self.density = density
        self.init_density = init_density
        self.style = style
        self.redshift = redshift

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        hdf5_path = row['file_path']

        if self.density:
            with h5py.File(hdf5_path, 'r') as f:
                # Load tensors and permute to match (C, D, H, W)
                density = torch.from_numpy(np.einsum('ijkc->cijk', f['density'][()])).float()
            if self.init_density:
                LPT = density[0]  # shape (1, 32, 32, 32)
                Nbody = density[1]  # shape (1, 32, 32, 32)
            else:
                LPT = density[3]
                Nbody = density[1]
            if self.augment:
                # Apply axis flips (no sign changes for density)
                if np.random.rand() < .5:
                    LPT = LPT[:, ::-1, ...]  # Flip x-axis
                    Nbody = Nbody[:, ::-1, ...]
                if np.random.rand() < .5:
                    LPT = LPT[:, :, ::-1, ...]  # Flip y-axis
                    Nbody = Nbody[:, :, ::-1, ...]
                if np.random.rand() < .5:
                    LPT = LPT[:, :, :, ::-1]  # Flip z-axis
                    Nbody = Nbody[:, :, :, ::-1]
                
                # Apply axis permutations (no component swaps needed)
                prand = np.random.rand()
                if prand < 1./6:
                    LPT = np.transpose(LPT, axes=(0, 2, 3, 1))  # Permute axes
                    Nbody = np.transpose(Nbody, axes=(0, 2, 3, 1))
                elif prand < 2./6:
                    LPT = np.transpose(LPT, axes=(0, 2, 1, 3))
                    Nbody = np.transpose(Nbody, axes=(0, 2, 1, 3))
                elif prand < 3./6:
                    LPT = np.transpose(LPT, axes=(0, 1, 3, 2))
                    Nbody = np.transpose(Nbody, axes=(0, 1, 3, 2))
                elif prand < 4./6:
                    LPT = np.transpose(LPT, axes=(0, 3, 1, 2))
                    Nbody = np.transpose(Nbody, axes=(0, 3, 1, 2))
                elif prand < 5./6:
                    LPT = np.transpose(LPT, axes=(0, 3, 2, 1))
                    Nbody = np.transpose(Nbody, axes=(0, 3, 2, 1))
        elif self.velocity:
            with h5py.File(hdf5_path, 'r') as f:
                velocity = torch.from_numpy(np.einsum('ijkc->cijk', f['velocity'][()])).float()
            LPT = velocity[6:9]
            Nbody = velocity[0:3]
        else:
            with h5py.File(hdf5_path, 'r') as f:
                displacement = torch.from_numpy(np.einsum('ijkc->cijk', f['displacement'][()])).float()
            LPT = displacement[6:9]
            Nbody = displacement[0:3]
        if self.augment and not self.density:
            if torch.rand(1).item() < 0.5:
                LPT = torch.flip(LPT, dims=[1])   # Reverse along dim=1
                LPT[0] = -LPT[0]
                Nbody = torch.flip(Nbody, dims=[1])
                Nbody[0] = -Nbody[0]

            if torch.rand(1).item() < 0.5:
                LPT = torch.flip(LPT, dims=[2])   # Reverse along dim=2
                LPT[1] = -LPT[1]
                Nbody = torch.flip(Nbody, dims=[2])
                Nbody[1] = -Nbody[1]

            if torch.rand(1).item() < 0.5:
                LPT = torch.flip(LPT, dims=[3])   # Reverse along dim=3
                LPT[2] = -LPT[2]
                Nbody = torch.flip(Nbody, dims=[3])
                Nbody[2] = -Nbody[2]

            prand = torch.rand(1).item()
            if prand < 1./6:
                LPT = LPT.permute(0, 2, 3, 1)
                LPT = swap_tensor(LPT, 0, 2)
                LPT = swap_tensor(LPT, 0, 1)
                Nbody = Nbody.permute(0, 2, 3, 1)
                Nbody = swap_tensor(Nbody, 0, 2)
                Nbody = swap_tensor(Nbody, 0, 1)
            elif prand < 2./6:
                LPT = LPT.permute(0, 2, 1, 3)
                LPT = swap_tensor(LPT, 0, 1)
                Nbody = Nbody.permute(0, 2, 1, 3)
                Nbody = swap_tensor(Nbody, 0, 1)
            elif prand < 3./6:
                LPT = LPT.permute(0, 1, 3, 2)
                LPT = swap_tensor(LPT, 1, 2)
                Nbody = Nbody.permute(0, 1, 3, 2)
                Nbody = swap_tensor(Nbody, 1, 2)
            elif prand < 4./6:
                LPT = LPT.permute(0, 3, 1, 2)
                LPT = swap_tensor(LPT, 1, 2)
                LPT = swap_tensor(LPT, 0, 1)
                Nbody = Nbody.permute(0, 3, 1, 2)
                Nbody = swap_tensor(Nbody, 1, 2)
                Nbody = swap_tensor(Nbody, 0, 1)
            elif prand < 5./6:
                LPT = LPT.permute(0, 3, 2, 1)
                LPT = swap_tensor(LPT, 0, 2)
                Nbody = Nbody.permute(0, 3, 2, 1)
                Nbody = swap_tensor(Nbody, 0, 2)

        if self.style is not None:
            style_values = torch.tensor(
                [row.get(k, 0.0) for k in self.style] + ([self.redshift] if self.redshift is not None else []),
                dtype=torch.float32
            )
            return LPT, Nbody, style_values

        return LPT, Nbody

class CSVHDF5DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_csv,
        val_csv,
        batch_size=32,
        num_workers=4,
        augment=True,
        velocity=False,
        density=False,
        init_density=False,
        style=None,
        redshift=None,
        shuffle=True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['kwargs'])

        self.train_csv = train_csv
        self.val_csv = val_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.velocity = velocity
        self.density = density
        self.init_density = init_density
        self.style = style
        self.redshift = redshift
        self.shuffle = shuffle

    def setup(self, stage=None):
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.batch_size = self.hparams.batch_size // dist.get_world_size()
        else:
            self.batch_size = self.hparams.batch_size
        self.train_dataset = HDF5AugmentedDataset(
            csv_file=self.train_csv,
            augment=self.augment,
            velocity=self.velocity,
            density=self.density,
            init_density=self.init_density,
            style=self.style,
            redshift=self.redshift,
        )
        self.val_dataset = HDF5AugmentedDataset(
            csv_file=self.val_csv,
            augment=False,
            velocity=self.velocity,
            density=self.density,
            init_density=self.init_density,
            style=self.style,
            redshift=self.redshift,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            #persistent_workers = self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )