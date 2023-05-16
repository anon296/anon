import numpy as np
import torch
import torch.utils.data

def _build_dsprites_relations(cfg, prior, digits_node_id, fix=None):
    def compute_rel1_target(start_idx):
        table = {
            "<^S":">^S", "<xS":">xS", "<vS":">vS", "x^S":"<^S", "xxS":"<xS", "xvS":"<vS", ">^S":"x^S", ">xS":"xxS", ">vS":"xvS",
            "<^E":">^E", "<xE":">xE", "<vE":">vE", "x^E":"<^E", "xxE":"<xE", "xvE":"<vE", ">^E":"x^E", ">xE":"xxE", ">vE":"xvE",
            "<^H":">^H", "<xH":">xH", "<vH":">vH", "x^H":"<^H", "xxH":"<xH", "xvH":"<vH", ">^H":"x^H", ">xH":"xxH", ">vH":"xvH",
        }
        start = cfg.data.class_names[start_idx]
        target = table[start]
        target_idx = cfg.data.class_names.index(target)
        return target_idx

    def compute_rel2_target(start_idx):
        table = {
            "<^S":"x^S", "<xS":"xxS", "<vS":"xvS", "x^S":">^S", "xxS":">xS", "xvS":">vS", ">^S":"<^S", ">xS":"<xS", ">vS":"<vS",
            "<^E":"x^E", "<xE":"xxE", "<vE":"xvE", "x^E":">^E", "xxE":">xE", "xvE":">vE", ">^E":"<^E", ">xE":"<xE", ">vE":"<vE",
            "<^H":"x^H", "<xH":"xxH", "<vH":"xvH", "x^H":">^H", "xxH":">xH", "xvH":">vH", ">^H":"<^H", ">xH":"<xH", ">vH":"<vH",
        }
        start = cfg.data.class_names[start_idx]
        target = table[start]

        target_idx = cfg.data.class_names.index(target)
        return target_idx

    def compute_rel3_target(start_idx):
        table = {
            "<^S":"<vS", "<xS":"<^S", "<vS":"<xS", "x^S":"xvS", "xxS":"x^S", "xvS":"xxS", ">^S":">vS", ">xS":">^S", ">vS":">xS",
            "<^E":"<vE", "<xE":"<^E", "<vE":"<xE", "x^E":"xvE", "xxE":"x^E", "xvE":"xxE", ">^E":">vE", ">xE":">^E", ">vE":">xE",
            "<^H":"<vH", "<xH":"<^H", "<vH":"<xH", "x^H":"xvH", "xxH":"x^H", "xvH":"xxH", ">^H":">vH", ">xH":">^H", ">vH":">xH",
        }
        start = cfg.data.class_names[start_idx]
        target = table[start]
        target_idx = cfg.data.class_names.index(target)
        return target_idx

    def compute_rel4_target(start_idx):
        table = {
            "<^S":"<xS", "<xS":"<vS", "<vS":"<^S", "x^S":"xxS", "xxS":"xvS", "xvS":"x^S", ">^S":">xS", ">xS":">vS", ">vS":">^S",
            "<^E":"<xE", "<xE":"<vE", "<vE":"<^E", "x^E":"xxE", "xxE":"xvE", "xvE":"x^E", ">^E":">xE", ">xE":">vE", ">vE":">^E",
            "<^H":"<xH", "<xH":"<vH", "<vH":"<^H", "x^H":"xxH", "xxH":"xvH", "xvH":"x^H", ">^H":">xH", ">xH":">vH", ">vH":">^H",
        }
        start = cfg.data.class_names[start_idx]
        target = table[start]
        target_idx = cfg.data.class_names.index(target)
        return target_idx

    def compute_rel5_target(start_idx):
        table = {
            "<^S":"<^E", "<xS":"<xE", "<vS":"<vE", "x^S":"x^E", "xxS":"xxE", "xvS":"xvE", ">^S":">^E", ">xS":">xE", ">vS":">vE",
            "<^E":"<^H", "<xE":"<xH", "<vE":"<vH", "x^E":"x^H", "xxE":"xxH", "xvE":"xvH", ">^E":">^H", ">xE":">xH", ">vE":">vH",
            "<^H":"<^S", "<xH":"<xS", "<vH":"<vS", "x^H":"x^S", "xxH":"xxS", "xvH":"xvS", ">^H":">^S", ">xH":">xS", ">vH":">vS"
        }
        start = cfg.data.class_names[start_idx]
        target = table[start]
        target_idx = cfg.data.class_names.index(target)
        return target_idx

    rel1 = np.random.choice(digits_node_id)
    rel1_sample = prior.gaussians[rel1].rsample()
    rel2 = np.random.choice(digits_node_id)
    rel2_sample = prior.gaussians[rel2].rsample()
    rel3 = np.random.choice(digits_node_id)
    rel3_sample = prior.gaussians[rel3].rsample()
    rel4 = np.random.choice(digits_node_id)
    rel4_sample = prior.gaussians[rel4].rsample()
    rel5 = np.random.choice(digits_node_id)
    rel5_sample = prior.gaussians[rel5].rsample()

    rel1_target = compute_rel1_target(rel1)
    rel1_target_sample = prior.gaussians[rel1_target].rsample()

    rel2_target = compute_rel2_target(rel2)
    rel2_target_sample = prior.gaussians[rel2_target].rsample()

    rel3_target = compute_rel3_target(rel3)
    rel3_target_sample = prior.gaussians[rel3_target].rsample()

    rel4_target = compute_rel4_target(rel4)
    rel4_target_sample = prior.gaussians[rel4_target].rsample()

    rel5_target = compute_rel5_target(rel5)
    rel5_target_sample = prior.gaussians[rel5_target].rsample()

    rel1_data = RelationalData(
        label=torch.tensor([rel1], dtype=torch.long),
        sample=rel1_sample,
        target_sample=rel1_target_sample,
        target_label=torch.tensor([rel1_target], dtype=torch.long)
    )
    rel2_data = RelationalData(
        label=torch.tensor([rel2], dtype=torch.long),
        sample=rel2_sample,
        target_sample=rel2_target_sample,
        target_label=torch.tensor([rel2_target], dtype=torch.long)
    )
    rel3_data = RelationalData(
        label=torch.tensor([rel3], dtype=torch.long),
        sample=rel3_sample,
        target_sample=rel3_target_sample,
        target_label=torch.tensor([rel3_target], dtype=torch.long)
    )
    rel4_data = RelationalData(
        label=torch.tensor([rel4], dtype=torch.long),
        sample=rel4_sample,
        target_sample=rel4_target_sample,
        target_label=torch.tensor([rel4_target], dtype=torch.long)
    )
    rel5_data = RelationalData(
        label=torch.tensor([rel5], dtype=torch.long),
        sample=rel5_sample,
        target_sample=rel5_target_sample,
        target_label=torch.tensor([rel5_target], dtype=torch.long)
    )
    return rel1_data, rel2_data, rel3_data, rel4_data, rel5_data

def _build_Shapes3D_relations(cfg, prior, digits_node_id, fix=None):
    def compute_rel1_target(start_idx):
        def table(start):
            hue, rel5, scale = start.split(",")
            hue_v   = int(hue[-1])
            rel5_v = int(rel5[-1])
            scale_v = int(scale[-1])

            if hue_v < 9:
                hue_v += 1

            return "HU" + str(hue_v) + ",SH" + str(rel5_v) + ",SC" + str(scale_v)

        start = cfg.data.class_names[start_idx]
        target = table(start)
        target_idx = cfg.data.class_names.index(target)
        return target_idx

    def compute_rel2_target(start_idx):

        def table(start):
            hue, rel5, scale = start.split(",")
            hue_v   = int(hue[-1])
            rel5_v = int(rel5[-1])
            scale_v = int(scale[-1])

            if hue_v > 0:
                hue_v -= 1

            return "HU" + str(hue_v) + ",SH" + str(rel5_v) + ",SC" + str(scale_v)

        start = cfg.data.class_names[start_idx]
        target = table[start]

        target_idx = cfg.data.class_names.index(target)
        return target_idx

    def compute_rel3_target(start_idx):
        def table(start):
            hue, rel5, scale = start.split(",")
            hue_v   = int(hue[-1])
            rel5_v = int(rel5[-1])
            scale_v = int(scale[-1])

            if scale_v < 2:
                scale_v += 1

            return "HU" + str(hue_v) + ",SH" + str(rel5_v) + ",SC" + str(scale_v)

        start = cfg.data.class_names[start_idx]
        target = table[start]
        target_idx = cfg.data.class_names.index(target)
        return target_idx


    def compute_rel4_target(start_idx):
        def table(start):
            hue, rel5, scale = start.split(",")
            hue_v   = int(hue[-1])
            rel5_v = int(rel5[-1])
            scale_v = int(scale[-1])

            if scale_v > 0:
                scale_v -= 1

            return "HU" + str(hue_v) + ",SH" + str(rel5_v) + ",SC" + str(scale_v)

        start = cfg.data.class_names[start_idx]
        target = table[start]
        target_idx = cfg.data.class_names.index(target)
        return target_idx


    def compute_rel5_target(start_idx):
        def table(start):
            hue, rel5, scale = start.split(",")
            hue_v   = int(hue[-1])
            rel5_v = int(rel5[-1])
            scale_v = int(scale[-1])

            if rel5_v < 4:
                rel5_v += 1
            else:
                rel5_v = 0

            return "HU" + str(hue_v) + ",SH" + str(rel5_v) + ",SC" + str(scale_v)

        start = cfg.data.class_names[start_idx]
        target = table[start]
        target_idx = cfg.data.class_names.index(target)
        return target_idx

    if fix is not None:
        assert type(fix) == int
        rel1 = fix
        rel1_sample = prior.gaussians[fix].rsample()
        rel2 = fix
        rel2_sample = prior.gaussians[fix].rsample()
        rel3 = fix
        rel3_sample = prior.gaussians[fix].rsample()
        rel4 = fix
        rel4_sample = prior.gaussians[fix].rsample()
        rel5 = fix
        rel5_sample = prior.gaussians[fix].rsample()
    else:
        rel1 = np.random.choice(digits_node_id)
        rel1_sample = prior.gaussians[rel1].rsample()
        rel2 = np.random.choice(digits_node_id)
        rel2_sample = prior.gaussians[rel2].rsample()
        rel3 = np.random.choice(digits_node_id)
        rel3_sample = prior.gaussians[rel3].rsample()
        rel4 = np.random.choice(digits_node_id)
        rel4_sample = prior.gaussians[rel4].rsample()
        rel5 = np.random.choice(digits_node_id)
        rel5_sample = prior.gaussians[rel5].rsample()

    rel1_target = compute_rel1_target(rel1)
    rel1_target_sample = prior.gaussians[rel1_target].rsample()

    rel2_target = compute_rel2_target(rel2)
    rel2_target_sample = prior.gaussians[rel2_target].rsample()

    rel3_target = compute_rel3_target(rel3)
    rel3_target_sample = prior.gaussians[rel3_target].rsample()

    rel4_target = compute_rel4_target(rel4)
    rel4_target_sample = prior.gaussians[rel4_target].rsample()

    rel5_target = compute_rel5_target(rel5)
    rel5_target_sample = prior.gaussians[rel5_target].rsample()

    rel1_data = RelationalData(
        label=torch.tensor([rel1], dtype=torch.long),
        sample=rel1_sample,
        target_sample=rel1_target_sample,
        target_label=torch.tensor([rel1_target], dtype=torch.long)
    )
    rel2_data = RelationalData(
        label=torch.tensor([rel2], dtype=torch.long),
        sample=rel2_sample,
        target_sample=rel2_target_sample,
        target_label=torch.tensor([rel2_target], dtype=torch.long)
    )
    rel3_data = RelationalData(
        label=torch.tensor([rel3], dtype=torch.long),
        sample=rel3_sample,
        target_sample=rel3_target_sample,
        target_label=torch.tensor([rel3_target], dtype=torch.long)
    )
    rel4_data = RelationalData(
        label=torch.tensor([rel4], dtype=torch.long),
        sample=rel4_sample,
        target_sample=rel4_target_sample,
        target_label=torch.tensor([rel4_target], dtype=torch.long)
    )
    rel5_data = RelationalData(
        label=torch.tensor([rel5], dtype=torch.long),
        sample=rel5_sample,
        target_sample=rel5_target_sample,
        target_label=torch.tensor([rel5_target], dtype=torch.long)
    )
    return rel1_data, rel2_data, rel3_data, rel4_data, rel5_data

def new_batch_of_relations(cfg, dset, batch_size, prior, digits_node_id, fix_rel1=None):
    relation1s = []
    relation2s = []
    relation3s = []
    relation4s = []
    relation5s = []

    if dset == 'dsprites':
        build_rels_func = _build_dsprites_relations
        nc = 1
    else:
        build_rels_func = _build_Shapes3D_relations
        nc = 3

    for _ in range(batch_size):
        lr, rr, ur, dr, sr = build_rels_func(cfg, prior, digits_node_id, fix_rel1)
        relation1s.append(lr)
        relation2s.append(rr)
        relation3s.append(ur)
        relation4s.append(dr)
        relation5s.append(sr)

    batch = RelationalBatch(nc, relation1s, relation2s, relation3s, relation4s, relation5s)
    return batch

class RelationalData():
    def __init__(self, label, sample, target_label, target_sample):
        self.label = label
        self.sample = sample
        self.target_sample = target_sample
        self.target_label = target_label

class RelationalBatch():
    def __init__(self, nc, rel1_relations, rel2_relations, rel3_relations, rel4_relations, rel5_relations):
        self.z_dim = nc

        self.rel1_label = []
        self.rel1_sample = []
        self.rel1_target_label = []
        self.rel1_target_sample = []

        for i, data in enumerate(rel1_relations):
            self.rel1_label.append(data.label)
            self.rel1_sample.append(data.sample)
            self.rel1_target_label.append(data.target_label)
            self.rel1_target_sample.append(data.target_sample)

        self.rel1_label = torch.cat(self.rel1_label, dim=0)
        self.rel1_sample = torch.vstack(self.rel1_sample)
        self.rel1_target_label = torch.cat(self.rel1_target_label, dim=0)
        self.rel1_target_sample = torch.vstack(self.rel1_target_sample)

        self.rel2_label = []
        self.rel2_sample = []
        self.rel2_target_label = []
        self.rel2_target_sample = []

        for i, data in enumerate(rel2_relations):
            self.rel2_label.append(data.label)
            self.rel2_sample.append(data.sample)
            self.rel2_target_label.append(data.target_label)
            self.rel2_target_sample.append(data.target_sample)

        self.rel2_label = torch.cat(self.rel2_label, dim=0)
        self.rel2_sample = torch.vstack(self.rel2_sample)
        self.rel2_target_label = torch.cat(self.rel2_target_label, dim=0)
        self.rel2_target_sample = torch.vstack(self.rel2_target_sample)

        self.rel3_label = []
        self.rel3_sample = []
        self.rel3_target_label = []
        self.rel3_target_sample = []

        for i, data in enumerate(rel3_relations):
            self.rel3_label.append(data.label)
            self.rel3_sample.append(data.sample)
            self.rel3_target_label.append(data.target_label)
            self.rel3_target_sample.append(data.target_sample)

        self.rel3_label = torch.cat(self.rel3_label, dim=0)
        self.rel3_sample = torch.vstack(self.rel3_sample)
        self.rel3_target_label = torch.cat(self.rel3_target_label, dim=0)
        self.rel3_target_sample = torch.vstack(self.rel3_target_sample)

        self.rel4_label = []
        self.rel4_sample = []
        self.rel4_target_label = []
        self.rel4_target_sample = []

        for i, data in enumerate(rel4_relations):
            self.rel4_label.append(data.label)
            self.rel4_sample.append(data.sample)
            self.rel4_target_label.append(data.target_label)
            self.rel4_target_sample.append(data.target_sample)

        self.rel4_label = torch.cat(self.rel4_label, dim=0)
        self.rel4_sample = torch.vstack(self.rel4_sample)
        self.rel4_target_label = torch.cat(self.rel4_target_label, dim=0)
        self.rel4_target_sample = torch.vstack(self.rel4_target_sample)

        self.rel5_label = []
        self.rel5_sample = []
        self.rel5_target_label = []
        self.rel5_target_sample = []

        for i, data in enumerate(rel5_relations):
            self.rel5_label.append(data.label)
            self.rel5_sample.append(data.sample)
            self.rel5_target_label.append(data.target_label)
            self.rel5_target_sample.append(data.target_sample)

        self.rel5_label = torch.cat(self.rel5_label, dim=0)
        self.rel5_sample = torch.vstack(self.rel5_sample)
        self.rel5_target_label = torch.cat(self.rel5_target_label, dim=0)
        self.rel5_target_sample = torch.vstack(self.rel5_target_sample)


    def to(self, device):
        self.rel1_label = self.rel1_label.to(device)
        self.rel1_sample = self.rel1_sample.to(device)
        self.rel1_target_label = self.rel1_target_label.to(device)
        self.rel1_target_sample = self.rel1_target_sample.to(device)

        self.rel2_label = self.rel2_label.to(device)
        self.rel2_sample = self.rel2_sample.to(device)
        self.rel2_target_label = self.rel2_target_label.to(device)
        self.rel2_target_sample = self.rel2_target_sample.to(device)

        self.rel3_label = self.rel3_label.to(device)
        self.rel3_sample = self.rel3_sample.to(device)
        self.rel3_target_label = self.rel3_target_label.to(device)
        self.rel3_target_sample = self.rel3_target_sample.to(device)

        self.rel4_label = self.rel4_label.to(device)
        self.rel4_sample = self.rel4_sample.to(device)
        self.rel4_target_label = self.rel4_target_label.to(device)
        self.rel4_target_sample = self.rel4_target_sample.to(device)

        self.rel5_label = self.rel5_label.to(device)
        self.rel5_sample = self.rel5_sample.to(device)
        self.rel5_target_label = self.rel5_target_label.to(device)
        self.rel5_target_sample = self.rel5_target_sample.to(device)

        return self
