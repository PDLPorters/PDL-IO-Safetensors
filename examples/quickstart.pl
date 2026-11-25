# Copyright 2026 Google LLC
#
# Licensed under the the same terms as Perl itself (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

use strict;
use warnings;
use File::Temp qw(tempfile);
use PDL;
use PDL::IO::Safetensors qw(rsafetensors wsafetensors safetensors_header);

my (undef, $path) = tempfile('pdl_st_XXXXXX', SUFFIX => '.safetensors', UNLINK => 1);

# 1. Save named PDL tensors and __metadata__ to a Safetensors file
my $tensors = {
    'linear.weight' => float([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    'linear.bias'   => float([0.1, -0.2]),
};

wsafetensors($path, $tensors, { format => 'pt', framework => 'pdl' });

# 2. Inspect the JSON header without loading tensor payloads
my $hdr = safetensors_header($path);
print 'Metadata format: ', $hdr->{__metadata__}{format}, "\n";
for my $name (sort grep { $_ ne '__metadata__' } keys %{$hdr}) {
    my $info = $hdr->{$name};
    printf "Tensor %-15s dtype=%-3s shape=[%s] bytes=%d\n",
        $name,
        $info->{dtype},
        join(',', @{ $info->{shape} }),
        $info->{data_offsets}[1] - $info->{data_offsets}[0];
}

# 3. Load tensors back into PDL ndarrays
my $loaded = rsafetensors($path);
my $w = $loaded->{'linear.weight'};
my $b = $loaded->{'linear.bias'};

printf "Loaded weight dims=(%s), bias sum=%.2f\n",
    join(',', $w->dims), $b->sum;
