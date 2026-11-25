# PDL::IO::Safetensors

Pure-Perl reader and writer for HuggingFace **Safetensors** (`.safetensors`) model weight files directly to and from [PDL](https://pdl.perl.org/) ndarrays.

## Quickstart

```perl
use strict;
use warnings;
use PDL;
use PDL::IO::Safetensors qw(rsafetensors wsafetensors safetensors_header);

# Save named PDL tensors and __metadata__
wsafetensors('weights.safetensors', {
    'linear.weight' => float([[1, 2, 3], [4, 5, 6]]),
    'linear.bias'   => float([0.1, -0.2]),
}, { format => 'pt' });

# Inspect JSON header without loading tensor payloads
my $hdr = safetensors_header('weights.safetensors');
print 'Format: ', $hdr->{__metadata__}{format}, "\n";

# Load tensors back into PDL
my $tensors = rsafetensors('weights.safetensors');
print $tensors->{'linear.weight'}, "\n";
```

See [examples/quickstart.pl](examples/quickstart.pl) for a complete runnable tutorial, and run `perldoc PDL::IO::Safetensors` for the full API reference and tutorial.

## Installation

```bash
perl Makefile.PL
make
make test
make install
```

## License

the same terms as Perl itself.
