package PDL::IO::Safetensors;

use 5.010;
use strict;
use warnings;
use Carp qw(croak);
use Exporter qw(import);
use JSON::MaybeXS qw(encode_json decode_json);
use PDL::Core ();
use PDL::Lite ();
use PDL::Types ();

our $VERSION   = '0.011';
our @EXPORT_OK = qw(rsafetensors wsafetensors safetensors_header);

my %DTYPE_TO_PDL = (
    F64 => PDL::double(),
    F32 => PDL::float(),
    I64 => PDL::longlong(),
    I32 => PDL::long(),
    I16 => PDL::short(),
    U8  => PDL::byte(),
    I8  => PDL::sbyte(),
);

my %PDL_TO_DTYPE = (
    double   => 'F64',
    float    => 'F32',
    longlong => 'I64',
    long     => 'I32',
    short    => 'I16',
    byte     => 'U8',
    sbyte    => 'I8',
);

sub safetensors_header {
    my ($file) = @_;
    croak 'Filename required' unless defined $file;
    open my $fh, '<:raw', $file or croak "Cannot open '$file': $!";
    my $len_buf;
    my $n = read($fh, $len_buf, 8);
    croak 'Truncated safetensors header length' unless defined $n && $n == 8;
    my $header_len = unpack('Q<', $len_buf);
    my $json_buf;
    $n = read($fh, $json_buf, $header_len);
    croak 'Truncated safetensors JSON header' unless defined $n && $n == $header_len;
    close $fh;
    return decode_json($json_buf);
}

sub rsafetensors {
    my ($file) = @_;
    croak 'Filename required' unless defined $file;
    open my $fh, '<:raw', $file or croak "Cannot open '$file': $!";
    my $len_buf;
    my $n = read($fh, $len_buf, 8);
    croak 'Truncated safetensors header length' unless defined $n && $n == 8;
    my $header_len = unpack('Q<', $len_buf);
    my $json_buf;
    $n = read($fh, $json_buf, $header_len);
    croak 'Truncated safetensors JSON header' unless defined $n && $n == $header_len;
    my $header = decode_json($json_buf);

    my $data_start = 8 + $header_len;
    my %tensors;

    for my $name (sort keys %{$header}) {
        next if $name eq '__metadata__';
        my $info = $header->{$name};
        my $dtype = $info->{dtype} // croak "Missing dtype for '$name'";
        my $pdl_type = $DTYPE_TO_PDL{$dtype}
            // croak "Unsupported safetensors dtype '$dtype' for '$name'";
        my $offsets = $info->{data_offsets} // croak "Missing data_offsets for '$name'";
        my ($start, $end) = @{$offsets};
        my $byte_len = $end - $start;

        seek($fh, $data_start + $start, 0) or croak "seek failed: $!";
        my $raw = '';
        if ($byte_len > 0) {
            my $got = read($fh, $raw, $byte_len);
            croak "Truncated tensor data for '$name'" unless defined $got && $got == $byte_len;
        }

        my @shape = reverse @{ $info->{shape} || [] };
        my $pdl = PDL->zeroes($pdl_type, @shape);
        ${ $pdl->get_dataref } = $raw;
        $pdl->upd_data;
        $tensors{$name} = $pdl;
    }

    close $fh;
    return \%tensors;
}

sub wsafetensors {
    my ($file, $tensors, $metadata) = @_;
    croak 'Filename and tensor hashref required'
        unless defined $file && ref($tensors) eq 'HASH';

    my %header;
    if (ref($metadata) eq 'HASH' && %{$metadata}) {
        $header{'__metadata__'} = $metadata;
    }

    my @chunks;
    my $offset = 0;

    for my $name (sort keys %{$tensors}) {
        my $pdl = $tensors->{$name};
        croak "Value for '$name' is not a PDL ndarray"
            unless ref($pdl) && eval { $pdl->isa('PDL') };

        my $type_name = $pdl->type->ioname;
        my $dtype = $PDL_TO_DTYPE{$type_name}
            // croak "Unsupported PDL type '$type_name' for '$name'";

        my $contig = $pdl->is_inplace ? $pdl : $pdl->make_physical;
        my $raw    = ${ $contig->get_dataref };
        my $len    = length($raw);
        my @shape  = reverse $contig->dims;

        $header{$name} = {
            dtype        => $dtype,
            shape        => \@shape,
            data_offsets => [ $offset, $offset + $len ],
        };
        push @chunks, $raw;
        $offset += $len;
    }

    my $json_bytes = encode_json(\%header);
    open my $fh, '>:raw', $file or croak "Cannot write '$file': $!";
    print {$fh} pack('Q<', length($json_bytes));
    print {$fh} $json_bytes;
    for my $chunk (@chunks) {
        print {$fh} $chunk;
    }
    close $fh or croak "Failed closing '$file': $!";
    return 1;
}

1;

__END__

=head1 NAME

PDL::IO::Safetensors - Read and write HuggingFace Safetensors files as PDL ndarrays

=head1 SYNOPSIS

    use PDL;
    use PDL::IO::Safetensors qw(rsafetensors wsafetensors safetensors_header);

    wsafetensors('weights.safetensors', {
        weight => float([[1, 2, 3], [4, 5, 6]]),
        bias   => float([0.1, 0.2]),
    }, { format => 'pt' });

    my $tensors = rsafetensors('weights.safetensors');
    my $header  = safetensors_header('weights.safetensors');

=head1 DESCRIPTION

C<PDL::IO::Safetensors> provides pure-Perl reading and writing of HuggingFace
C<.safetensors> model weight files directly to and from L<PDL> ndarrays.

=head1 FUNCTIONS

=head2 wsafetensors($file, \%tensors, [\%metadata])

Writes a hashref of named L<PDL> ndarrays to C<$file> in Safetensors format.

=head2 rsafetensors($file)

Reads all tensors from C<$file> and returns a hashref mapping tensor names to
L<PDL> ndarrays.

=head2 safetensors_header($file)

Reads and returns only the decoded JSON header from C<$file>.

=head1 TUTORIAL

This tutorial translates the HuggingFace Safetensors save/load quickstart into
idiomatic Perl and L<PDL>.

=head2 1. Save Named PDL Tensors to a Safetensors File

Pass the output path, a hashref of named L<PDL> ndarrays, and an optional
C<__metadata__> hashref to C<wsafetensors>:

    use strict;
    use warnings;
    use File::Temp qw(tempfile);
    use PDL;
    use PDL::IO::Safetensors qw(rsafetensors wsafetensors safetensors_header);

    my (undef, $path) = tempfile(SUFFIX => '.safetensors', UNLINK => 1);

    my $tensors = {
        'linear.weight' => float([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        'linear.bias'   => float([0.1, -0.2]),
    };

    wsafetensors($path, $tensors, { format => 'pt', framework => 'pdl' });

=head2 2. Inspect the JSON Header Without Loading Tensor Data

Use C<safetensors_header> to inspect metadata, dtypes, row-major shapes, and
byte offsets without reading tensor payloads:

    my $hdr = safetensors_header($path);
    print 'Format: ', $hdr->{__metadata__}{format}, "\n";
    for my $name (sort grep { $_ ne '__metadata__' } keys %{$hdr}) {
        my $info = $hdr->{$name};
        printf "Tensor %-15s dtype=%-3s shape=[%s] bytes=%d\n",
            $name,
            $info->{dtype},
            join(',', @{ $info->{shape} }),
            $info->{data_offsets}[1] - $info->{data_offsets}[0];
    }

=head2 3. Load Tensors Back into PDL

Use C<rsafetensors> to load all tensors into a hashref of L<PDL> ndarrays and
perform vectorized operations:

    my $loaded = rsafetensors($path);
    my $w = $loaded->{'linear.weight'};
    my $b = $loaded->{'linear.bias'};

    printf "Weight dims: (%s), bias sum: %.2f\n",
        join(',', $w->dims), $b->sum;

See C<examples/quickstart.pl> in the distribution for a complete runnable
script.

=head1 TYPES

Supports real numeric tensor dtypes defined by the Safetensors specification
(C<F32>/C<float>, C<F64>/C<double>, C<I8>/C<sbyte>, C<U8>/C<byte>,
C<I16>/C<short>, C<U16>/C<ushort>, C<I32>/C<long>, C<U32>/C<ulong>,
C<I64>/C<longlong>, C<U64>/C<ulonglong>). Complex types are not part of the
Safetensors format specification.

=head1 AUTHOR

C.J. Collier E<lt>cjac@google.comE<gt>

=head1 COPYRIGHT AND LICENSE

Copyright 2026 Google LLC

This library is free software; you can redistribute it and/or modify it under
the same terms as Perl itself.

=cut
