from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
print("works!")
# since no GPU the rest should not work...
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-2.8b-slimpj")