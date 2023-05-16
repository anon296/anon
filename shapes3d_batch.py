
class Shapes3D_RelationalBatch():
    def __init__(self, cfg, left_relations, right_relations, up_relations, down_relations, shape_relations):
        super(Shapes3D_RelationalBatch, self).__init__()
        self.z_dim = cfg.model.z_dim

        self.left_label = []
        self.left_sample = []
        self.left_target_label = []
        self.left_target_sample = []

        for i, data in enumerate(left_relations):
            self.left_label.append(data.label)
            self.left_sample.append(data.sample)
            self.left_target_label.append(data.target_label)
            self.left_target_sample.append(data.target_sample)

        self.left_label = torch.cat(self.left_label, dim=0)
        self.left_sample = torch.vstack(self.left_sample)
        self.left_target_label = torch.cat(self.left_target_label, dim=0)
        self.left_target_sample = torch.vstack(self.left_target_sample)

        self.right_label = []
        self.right_sample = []
        self.right_target_label = []
        self.right_target_sample = []

        for i, data in enumerate(right_relations):
            self.right_label.append(data.label)
            self.right_sample.append(data.sample)
            self.right_target_label.append(data.target_label)
            self.right_target_sample.append(data.target_sample)

        self.right_label = torch.cat(self.right_label, dim=0)
        self.right_sample = torch.vstack(self.right_sample)
        self.right_target_label = torch.cat(self.right_target_label, dim=0)
        self.right_target_sample = torch.vstack(self.right_target_sample)

        self.up_label = []
        self.up_sample = []
        self.up_target_label = []
        self.up_target_sample = []

        for i, data in enumerate(up_relations):
            self.up_label.append(data.label)
            self.up_sample.append(data.sample)
            self.up_target_label.append(data.target_label)
            self.up_target_sample.append(data.target_sample)

        self.up_label = torch.cat(self.up_label, dim=0)
        self.up_sample = torch.vstack(self.up_sample)
        self.up_target_label = torch.cat(self.up_target_label, dim=0)
        self.up_target_sample = torch.vstack(self.up_target_sample)

        self.down_label = []
        self.down_sample = []
        self.down_target_label = []
        self.down_target_sample = []

        for i, data in enumerate(down_relations):
            self.down_label.append(data.label)
            self.down_sample.append(data.sample)
            self.down_target_label.append(data.target_label)
            self.down_target_sample.append(data.target_sample)

        self.down_label = torch.cat(self.down_label, dim=0)
        self.down_sample = torch.vstack(self.down_sample)
        self.down_target_label = torch.cat(self.down_target_label, dim=0)
        self.down_target_sample = torch.vstack(self.down_target_sample)

        self.shape_label = []
        self.shape_sample = []
        self.shape_target_label = []
        self.shape_target_sample = []

        for i, data in enumerate(shape_relations):
            self.shape_label.append(data.label)
            self.shape_sample.append(data.sample)
            self.shape_target_label.append(data.target_label)
            self.shape_target_sample.append(data.target_sample)

        self.shape_label = torch.cat(self.shape_label, dim=0)
        self.shape_sample = torch.vstack(self.shape_sample)
        self.shape_target_label = torch.cat(self.shape_target_label, dim=0)
        self.shape_target_sample = torch.vstack(self.shape_target_sample)


    def to(self, device):
        self.left_label = self.left_label.to(device)
        self.left_sample = self.left_sample.to(device)
        self.left_target_label = self.left_target_label.to(device)
        self.left_target_sample = self.left_target_sample.to(device)

        self.right_label = self.right_label.to(device)
        self.right_sample = self.right_sample.to(device)
        self.right_target_label = self.right_target_label.to(device)
        self.right_target_sample = self.right_target_sample.to(device)

        self.up_label = self.up_label.to(device)
        self.up_sample = self.up_sample.to(device)
        self.up_target_label = self.up_target_label.to(device)
        self.up_target_sample = self.up_target_sample.to(device)

        self.down_label = self.down_label.to(device)
        self.down_sample = self.down_sample.to(device)
        self.down_target_label = self.down_target_label.to(device)
        self.down_target_sample = self.down_target_sample.to(device)

        self.shape_label = self.shape_label.to(device)
        self.shape_sample = self.shape_sample.to(device)
        self.shape_target_label = self.shape_target_label.to(device)
        self.shape_target_sample = self.shape_target_sample.to(device)

        return self
