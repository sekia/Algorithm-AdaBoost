package Algorithm::AdaBoost::Classifier;

use strict;
use warnings;
use v5.10;
use overload '&{}' => \&as_closure;
use Algorithm::AdaBoost::Util qw/assert_no_rest_params/;
use Carp qw//;
use List::Util qw//;

sub new {
  my ($class, %params) = @_;

  my $weak_classifiers = delete $params{weak_classifiers}
    // Carp::croak('Missing mandatory parameter: "weak_classifiers"');
  assert_no_rest_params %params;

  bless +{ weak_classifiers => $weak_classifiers } => $class;
}

sub as_closure {
  my ($self) = @_;

  return sub { $self->classify(@_) };
}

sub classify {
  my ($self, $feature) = @_;

  List::Util::sum(
    map {
      $_->{weight} * $_->{classifier}->($feature);
    } @{ $self->{weak_classifiers} }
  );
}

1;
__END__

=head1 NAME

Algorithm::AdaBoost::Classifier

=head1 DESCRIPTION

This class should be instanciated via C<< Algorithm::AdaBoost->train >>.

=head1 METHODS

=head2 as_closure

Returns a CodeRef which delegates given arguments to C<classify>.

Altough you can use the object itself like a CodeRef because C<&{}> operator is overloaded with this method, it constructs a closure for each call.
So if you classify many inputs, you should hold a closure explicitly or use C<classify> directly.

=head2 classify

Executes binary classification. Takes 1 argument as a feature and return a real number. If the number is positive, given feature is considered to belong to one class. Similary, If the number is negative, given feature is considered to belong to another one class. If zero is returned, it means that the feature cannot be classified (very rare case).

=head1 AUTHOR

Koichi SATOH E<lt>sekia@cpan.orgE<gt>

=cut
