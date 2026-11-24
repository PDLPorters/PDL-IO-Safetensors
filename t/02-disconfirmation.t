use strict;
use warnings;
use Test::More;
use File::Temp qw(tempfile);
use PDL::IO::Safetensors qw(rsafetensors safetensors_header);

subtest 'disconfirmation: missing file' => sub {
    my $ok = eval { rsafetensors('/nonexistent/file.safetensors'); 1 };
    ok(!$ok, 'rsafetensors dies on missing file');
};

subtest 'disconfirmation: corrupt JSON header' => sub {
    my ($fh, $fname) = tempfile(UNLINK => 1, SUFFIX => '.safetensors');
    binmode($fh);
    my $bad = '{not valid json';
    print {$fh} pack('Q<', length($bad)) . $bad;
    close($fh);

    my $ok = eval { safetensors_header($fname); 1 };
    ok(!$ok, 'safetensors_header rejects malformed JSON');
};

done_testing();
