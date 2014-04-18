package Algorithm::AdaBoost;

# ABSTRACT: AdaBoost meta-learning algorithm

use strict;
use warnings;
use v5.10;
use Algorithm::AdaBoost::Classifier;
use Algorithm::AdaBoost::Util qw/assert_no_rest_params/;
use Carp qw//;
use List::Util qw//;

our $VERSION = '0.01';

sub new {
  my ($class, %params) = @_;

  my %self = map {
    exists $params{$_} ? ($_ => delete $params{$_}) : ()
  } qw/training_set weak_classifier_generator/;
  assert_no_rest_params %params;

  bless \%self => $class;
}

sub calculate_classifier_weight {
  my ($self, %params) = @_;

  my $classifier = delete $params{classifier}
    // Carp::croak('Missing mandatory parameter: "classifier"');
  my $distribution = delete $params{distribution}
    // Carp::croak('Missing mandatory parameter: "distribution"');
  my $training_set = delete $params{training_set}
    // Carp::croak('Missing mandatory parameter: "training_set"');
  assert_no_rest_params %params;

  my $error_ratio = $self->evaluate_error_ratio(
    classifier => $classifier,
    distribution => $distribution,
    training_set => $training_set,
  );
  return log((1 - $error_ratio) / $error_ratio) / 2;
}

sub classify {
  my ($self, $feature) = @_;

  Carp::croak 'Training phase is undone yet.' unless $self->trained;
  $self->final_classifier->classify($feature);
}

sub construct_hardest_distribution {
  my ($self, %params) = @_;

  my $classifier = delete $params{classifier}
    // Carp::croak('Missing mandatory parameter: "classifier"');
  my $previous_distribution = delete $params{previous_distribution}
    // Carp::croak('Missing mandatory parameter: "previous_distribution"');
  my $training_set = delete $params{training_set}
    // Carp::croak('Missing mandatory parameter: "training_set"');
  my $weight = delete $params{weight}
    // Carp::croak('Missing mandatory parameter: "weight"');
  assert_no_rest_params %params;

  my @distribution = map {
    my $training_data = $training_set->[$_];
    $previous_distribution->[$_]
      * exp(-$weight * $training_data->{label}
              * $classifier->($training_data->{feature}));
  } 0 .. $#$previous_distribution;
  my $partition_function = List::Util::sum(@distribution);
  [ map { $_ / $partition_function } @distribution ];
}

sub evaluate_error_ratio {
  my ($self, %params) = @_;

  my $classifier = delete $params{classifier}
    // Carp::croak('Missing mandatory parameter: "classifier"');
  my $distribution = delete $params{distribution}
    // Carp::croak('Missing mandatory parameter: "distribution"');
  my $training_set = delete $params{training_set}
    // Carp::croak('Missing mandatory parameter: "training_set"');
  assert_no_rest_params %params;

  my $accuracy = 0;
  for my $i (0 .. $#$distribution) {
    my $training_data = $training_set->[$i];
    if ($classifier->($training_data->{feature}) == $training_data->{label}) {
      $accuracy += $distribution->[$i];
    }
  }
  return 1 - $accuracy;
}

sub final_classifier {
  my ($self) = @_;

  Carp::croak 'The classifier is not trained' unless $self->trained;
  return $self->{final_classifier};
}

sub train {
  my ($self, %params) = @_;

  my $num_iterations = delete $params{num_iterations}
    // Carp::croak('Missing mandatory parameter: "num_iterations"');
  my $training_set = delete $params{training_set}
    // $self->training_set
    // Carp::croak('Given no training set.');
  my $weak_classifier_generator = delete $params{weak_classifier_generator}
    // $self->weak_classifier_generator
    // Carp::croak('Given no weak classifier generator.');
  assert_no_rest_params %params;

  my $num_training_set = @$training_set;

  # Initial distribution is uniform.
  my $distribution = [ (1 / $num_training_set) x $num_training_set ];

  my ($weak_classifier, $weight);
  my @weak_classifiers;
  while ($num_iterations--) {
    # Construct a weak classifier which classifies data on the distribution.
    $weak_classifier = $weak_classifier_generator->(
      distribution => $distribution,
      training_set => $training_set,
    );
    $weight = $self->calculate_classifier_weight(
      classifier => $weak_classifier,
      distribution => $distribution,
      training_set => $training_set,
    );
    push @weak_classifiers, +{
      classifier => $weak_classifier,
      weight => $weight,
    };
  } continue {
    $distribution = $self->construct_hardest_distribution(
      classifier => $weak_classifier,
      previous_distribution => $distribution,
      training_set => $training_set,
      weight => $weight,
    );
  }

  return $self->{final_classifier} = Algorithm::AdaBoost::Classifier->new(
    weak_classifiers => \@weak_classifiers,
  );
}

sub trained { exists $_[0]->{final_classifier} }

sub training_set { $_[0]->{training_set} }

sub weak_classifier_generator { $_[0]->{weak_classifier_generator} }

1;
__END__

=head1 SYNOPSIS

  use Algorithm::AdaBoost;

  # Training phase.
  my $learner = Alogrithm::AdaBoost->new(
    training_set => [
      +{ feature => [...], label => 1, },
      +{ feature => [...], label => -1, },
      +{ feature => [...], label => -1, },
      ...
    ],
    weak_classifier_generator => \&my_poor_learning_algorithm,
  );
  $learner->train(num_iterations => 1_000);

  # Now you have a boost-ed classifier (Algorithm::AdaBoost::Classifier).
  my $classifier = $learner->final_classifier;
  given ($classifier->classify([...])) {
    when ($_ > 0) { say 'The data belongs to class 1.' }
    when ($_ < 0) { say 'The data belongs to class 2.' }
    default { warn 'The data cannot be classified.' }
  }

=head1 DESCRIPTION

AdaBoost is a machine learning algorithm proposed by Freund and Schapire.
Using an arbitrary binary classification algorithm, The algorithm can construct a more accurate classifier (i.e. it is a meta-algorithm).

=head1 METHODS

=head2 new

Constructor. You can specify 2 optional attributes:

=over 2

=item training_set

An ArrayRef which is used as a training data set.

Each item is a HashRef having 2 keys: C<feature> and C<label>. C<feature> is a arbitrary input that classifier accepts and C<label> is a expected output label (C<+1> or C<-1>).

=item weak_classifier_generator

A CodeRef which is expected to generate a binary classifier function.

When the function is called, 2 named parameters are specified like this:

  my $classifier = $generator->(
     distribution => [...],
     training_set => [...],
  );

C<distribution> is an ArrayRef which each item is a probability of corresponding item in C<training_set>. i.e. C<distribution> is P(X = t_i) where t_i is i-th item in C<training_set>.

The generated classifier is expected to be a CodeRef which takes 1 argument (value of C<feature>) and return C<+1> or C<-1> as a output label.

=back

Either of both can be overriden temporarily with parameters for C<train>.

=head2 classify

Shorthand for C<< $learner->final_classifier->classify >>.

=head2 final_classifier

Returns the last constructed classifier.

=head2 train

Constructs a stronger classifier from given training set and weak learning algorithm.

This method takes 1 mandatory parameter:

=over 2

=item num_iterations

Specifies how many training iterations to be excuted (i.e., how many weak classifiers to be generated).

=back

and 2 optional parameters:

=over 2

=item training_set

=item weak_classifier_generator

=back

If the optional parameters are ommited, parameters specified to C<new> are used as defaults. If constructor parameters are ommited too, an exception will be raised.

=head2 trained

True if C<train> method have called, false otherwise.

=head1 AUTHOR

Koichi SATOH E<lt>sekia@cpan.orgE<gt>

=head1 SEE ALSO

L<A Short Introduction to Boosting|http://www.site.uottawa.ca/~stan/csi5387/boost-tut-ppr.pdf>

=cut
