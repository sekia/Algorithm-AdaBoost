package Algorithm::AdaBoost::Util;

use strict;
use warnings;
use v5.10;
use Carp qw//;

our @CARP_NOT;

sub import {
  my ($class, @symbols) = @_;

  my $caller = caller;
  for my $symbol (@symbols) {
    no strict 'refs';
    Carp::croak(qq/Not exportable function: "$symbol"/) unless defined &$symbol;
    *{ "${caller}::${symbol}" } = \&$symbol;
  }
}

sub assert_no_rest_params(\%) {
  my ($params) = @_;

  if (%$params) {
    local @CARP_NOT = (scalar caller);
    Carp::croak(
      'Unknown parameter(s): ',
      join ', ', map { qq/"$_"/ } keys %$params,
    );
  }
}

1;
