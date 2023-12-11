from .dataset_factory import DatasetFactory
from .empty import EmptyDataset
from .inv3d import Inv3DDataset, Inv3DRealUnwarpDataset, Inv3DTestDataset

# training datasets
DatasetFactory.register_dataset("empty", EmptyDataset)
DatasetFactory.register_dataset("inv3d", Inv3DDataset)
DatasetFactory.register_dataset("inv3d_real_unwarp", Inv3DRealUnwarpDataset)

for unwarp_factor in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    DatasetFactory.register_dataset(
        name=f"inv3d_test--alpha={unwarp_factor}",
        dataset_class=Inv3DTestDataset,
        unwarp_factor=unwarp_factor,
        limit_samples=None,
    )
