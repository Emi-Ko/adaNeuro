import numpy as np
from scipy.ndimage.filters import gaussian_filter

from .preprocess import check_tensor

class BlurAugment():
    """
    Introduce out-of-focus section(s) to a training example. The number of
    out-of-focus sections to introduce is randomly drawn from the uniform
    distribution between [0, MAX_SEC]. Default MAX_SEC is 1, which can be
    overwritten by user-specified value.
    Out-of-focus process is implemented with Gaussian blurring.
    """

    def __init__(self, max_sec=1, skip_ratio=0.3, mode='full', sigma_max=5.0):
        """Initialize BlurAugment."""
        self.set_max_sections(max_sec)
        self.set_skip_ratio(skip_ratio)
        self.set_mode(mode)
        self.set_sigma_max(sigma_max)

    def prepare(self, x):
        return x

    def augment(self, img):
        """Apply out-of-focus section data augmentation."""
        if np.random.rand() > self.skip_ratio:
            img = self._do_augment(img)
        return img

    def _do_augment(self, img):
        """Apply out-of-section section data augmentation."""
        # Randomly draw the number of sections to introduce.
        num_sec = np.random.randint(1, self.MAX_SEC + 1)
        zdim = img.shape[-3]
        # Randomly draw z-slices to blur.
        zlocs = np.random.choice(zdim, num_sec, replace=False)
        # Apply full or partial missing sections according to the mode.
        print(zlocs)
        for z in zlocs:
            sigma = np.random.rand() * self.sigma_max
            img[...,z,:,:] = gaussian_filter(img[...,z,:,:], sigma=sigma)
        return img

    def set_max_sections(self, max_sec):
        """Set the maximum number of missing sections to introduce."""
        assert max_sec >= 0
        self.MAX_SEC = max_sec

    def set_skip_ratio(self, ratio):
        """Set the probability of skipping augmentation."""
        assert ratio >= 0.0 and ratio <= 1.0
        self.skip_ratio = ratio

    def set_mode(self, mode):
        """Set full/partial/mix missing section mode."""
        assert mode=='full' or mode=='partial' or mode=='mix'
        self.mode = mode

    def set_sigma_max(self, sigma_max):
        """Set the maximum sigma of the Gaussian blur filter."""
        assert sigma_max >= 0
        self.sigma_max = sigma_max

class MissingAugment():
    """
        Missing section data augmentation.
        Introduce missing section(s) to a training example. The number of missing
        sections to introduce is randomly drawn from the uniform distribution
        between [0, MAX_SEC]. Default MAX_SEC is 1, which can be overwritten by
        a user-specified value.
    """

    def __init__(self, max_sec=1, skip_ratio=0.3, mode='full',
                 consecutive=False, random_color=False):
        """Initialize MissingSectionAugment."""
        self.set_max_sections(max_sec)
        self.set_skip_ratio(skip_ratio)
        self.set_mode(mode)
        self.consecutive = consecutive
        self.random_color = random_color

        # DEBUG(kisuk)
        # self.hist = [0] * (max_sec + 1)

    def set_max_sections(self, max_sec):
        """Set the maximum number of missing sections to introduce."""
        assert max_sec >= 0
        self.MAX_SEC = max_sec

    def set_skip_ratio(self, ratio):
        """Set the probability of skipping augmentation."""
        assert ratio >= 0.0 and ratio <= 1.0
        self.skip_ratio = ratio

    def set_mode(self, mode):
        """Set full/partial/mix missing section mode."""
        assert mode=='full' or mode=='partial' or mode=='mix'
        self.mode = mode

    def prepare(self, x):
        """
            No change in spec.
        """
        return x

    def augment(self, img):
        """Apply missing section data augmentation."""
        if np.random.rand() > self.skip_ratio:
            img = self._do_augment(img)
        return img

    def _do_augment(self, img):
        """Apply missing section data augmentation."""
        num_sec = np.random.randint(1, self.MAX_SEC + 1)
        zdim = img.shape[-3]
        if self.consecutive:
            zloc  = np.random.randint(0, zdim - num_sec + 1)
            zlocs = range(zloc, zloc + num_sec)
        else:
            zlocs = np.random.choice(zdim, num_sec, replace=False)
        print("Missing", zlocs)
        # Fill-out value.
        val = np.random.rand() if self.random_color else 0
        if self.mode == 'full':
            img[...,zlocs,:,:] = val
        else:
            # Draw a random xy-coordinate.
            x = np.random.randint(0, xdim)
            y = np.random.randint(0, ydim)
            rule = np.random.rand(4) > 0.5

            for z in zlocs:
                val = np.random.rand() if self.random_color else 0
                if self.mode == 'mix' and np.random.rand() > 0.5:
                    img[...,zlocs,:,:] = val
                else:
                    # Independent coordinates across sections.
                    if not self.consecutive:
                        x = np.random.randint(0, xdim)
                        y = np.random.randint(0, ydim)
                        rule = np.random.rand(4) > 0.5
                    # 1st quadrant.
                    if rule[0]:
                        img[...,zlocs,:,:] = val
                    # 2nd quadrant.
                    if rule[1]:
                        img[...,zlocs,:,:] = val
                    # 3nd quadrant.
                    if rule[2]:
                        img[...,zlocs,:,:] = val
                    # 4nd quadrant.
                    if rule[3]:
                        img[...,zlocs,:,:] = val
        return img
    
class MisalignAugment():
    """
        Misalignment data augmentation.
    """
    def __init__(self, max_trans=8.0, slip_ratio=0.3, skip_ratio=0.0):
        self.set_max_translation(max_trans)
        self.set_slip_ratio(slip_ratio)
        self.set_skip_ratio(skip_ratio)

    def prepare(self, fov):
        self.skip = np.random.rand() < self.skip_ratio
        
        if not self.skip:
            self.spec = fov
            max_trans = self.max_trans
            self.x_t = int(round(max_trans * np.random.rand(1)[0])) * 2
            self.y_t = int(round(max_trans * np.random.rand(1)[0])) * 2
            
            z, y, x = fov[-3:]
            assert z > 0
            x_dim  = self.x_t + x
            y_dim  = self.y_t + y
            infov = fov[:-2] + (y_dim, x_dim)
            
            # Random direction of translation.
            x_sign = np.random.choice(['+','-'])
            y_sign = np.random.choice(['+','-'])
            self.x_t = int(eval(x_sign + str(self.x_t)))
            self.y_t = int(eval(y_sign + str(self.y_t)))

            # Introduce misalignment at pivot.
            self.pivot = np.random.randint(1, z - 1)
            self.slip = np.random.rand() < self.slip_ratio
            
        return np.array(infov)

    def augment(self, img, lbl):
        if not self.skip:
            img, lbl = self.do_augment(img, lbl)
        return img, lbl

    def do_augment(self, img, lbl):
        z, y, x = img.shape[-3:]
        assert z > 1
        
        # Ensure data is a 4D tensor.
        img = check_tensor(img)
        lbl = check_tensor(lbl)

        new_img = np.zeros(self.spec, dtype=img.dtype)
        new_img = check_tensor(new_img)

        new_lbl = np.zeros(self.spec, dtype=lbl.dtype)
        new_lbl = check_tensor(new_lbl)

        if self.slip:
            # Copy whole box.
            xmin = max(self.x_t, 0)
            ymin = max(self.y_t, 0)
            xmax = min(self.x_t, 0) + x
            ymax = min(self.y_t, 0) + y
            new_lbl[:,:,...] = lbl[:,:,ymin:ymax,xmin:xmax]
            new_img[:,:,...] = img[:,:,ymin:ymax,xmin:xmax]
            # Slip.
            xmin = max(-self.x_t, 0)
            ymin = max(-self.y_t, 0)
            xmax = min(-self.x_t, 0) + x
            ymax = min(-self.y_t, 0) + y
            pvot = self.pivot
            new_lbl[:,pvot,...] = lbl[:,pvot,ymin:ymax,xmin:xmax]
            new_img[:,pvot,...] = img[:,pvot,ymin:ymax,xmin:xmax]
        else:
            # Copy upper box.
            xmin = max(self.x_t, 0)
            ymin = max(self.y_t, 0)
            xmax = min(self.x_t, 0) + x
            ymax = min(self.y_t, 0) + y
            pvot = self.pivot
            new_lbl[:,0:pvot,...] = lbl[:,0:pvot,ymin:ymax,xmin:xmax]
            new_img[:,0:pvot,...] = img[:,0:pvot,ymin:ymax,xmin:xmax]
            # Copy lower box.
            xmin = max(-self.x_t, 0)
            ymin = max(-self.y_t, 0)
            xmax = min(-self.x_t, 0) + x
            ymax = min(-self.y_t, 0) + y
            pvot = self.pivot
            new_lbl[:, pvot:,...] = lbl[:,pvot:,ymin:ymax,xmin:xmax]
            new_img[:, pvot:,...] = img[:,pvot:,ymin:ymax,xmin:xmax]
            
        return new_img, new_lbl

    def set_max_translation(self, max_trans):
        """
            Set the maximum amount of translation in x and y
        """
        assert max_trans > 0
        self.max_trans = max_trans

    def set_slip_ratio(self, ratio):
        """
            How often is slip miaslignment introduced?
        """
        assert ratio >= 0.0 and ratio <= 1.0
        self.slip_ratio = ratio

    def set_skip_ratio(self, ratio):
        """
            Set the probability of skipping augmentation.
        """
        assert ratio >= 0.0 and ratio <= 1.0
        self.skip_ratio = ratio