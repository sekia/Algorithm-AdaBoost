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

  my $error_ratio_threshold = delete $params{error_ratio_threshold} // 0.50;
  my $num_iterations = delete $params{num_iterations} // 0+'inf';
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
    my $error_ratio = $self->evaluate_error_ratio(
      classifier => $weak_classifier,
      distribution => $distribution,
      training_set => $training_set,
    );
    last if $error_ratio >= $error_ratio_threshold;
    $weight = log((1 - $error_ratio) / $error_ratio) / 2;
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
      # Structure of |feature| is arbitrary. |label| must be +1 or -1.
      +{ feature => [...], label => 1, },
      +{ feature => [...], label => -1, },
      +{ feature => [...], label => -1, },
      ...
    ],
    weak_classifier_generator => \&my_poor_learning_algorithm,
  );
  $learner->train;

  # Now you have a boost-ed classifier (Algorithm::AdaBoost::Classifier).
  my $classifier = $learner->final_classifier;
  my $result = $classifier->classify([...]);
  if ($result > 0) {
    say 'Positive sample.';
  } elsif ($result < 0) {
    say 'Negative sample.';
  } else {
    warn 'Data cannot be classified.'
  }

=head1 DESCRIPTION

AdaBoost is a machine learning algorithm proposed by Freund and Schapire.
Using an arbitrary binary classification algorithm, The algorithm can construct a more accurate classifier (i.e. it is a meta-algorithm).

=head1 METHODS

=head2 new([training_set => \@training_set] [, weak_classifier_generator => \&weak_classifier_generator])

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

=head2 classify($feature)

Shorthand for C<< $learner->final_classifier->classify >>.

=head2 final_classifier

Returns the last C<train>ed classifier.

Note that the classifier has no dependnecy on C<Algorithm::AdaBoost> (i.e., learner) instance. So you can store the classifier in other place and reuse the learner for training other classifiers, or C<undef> the learner for freeing up memory.

=head2 train([error_ratio_threshold => 0.50] [, num_iterations => 0+'inf'] [, training_set => \@training_set] [, weak_classifier_generator => \&weak_classifier_generator])

Constructs an boosted classifier from given training set and weak learning algorithm.

=over 2

=item error_ratio_threshold

Criterion for stopping training iteration. The iteration will stop if a generated weak classifier's error ratio is not less than this value, regardless of the value of C<num_iterations>.

By default its value is 0.50. So the iteration will stop if the given learning algorithm could generate weak classifier accurate than random guess no more.

=item num_iterations

Specifies how many training iterations to be excuted (i.e., how many weak classifiers to be generated) at most. If ommited, the iteration will continue until the learning algorithm emit a weak classifier having error ratio not less than C<error_ratio_threshold>.

Summarizing up, the training iteration will stop when either or both conditions below is satisfied:

=over 2

=item 1.

The given learning algorithm generated a classifier and its error ratio is equal to or worse than C<error_ratio_threshold>.

=item 2.

The iteration is executed C<num_iterations> times.

=back

=item training_set

=item weak_classifier_generator

These 2 options are same as constuctor parameters.

=back

If the optional parameters are ommited, parameters specified to C<new> are used as defaults. If constructor parameters are ommited too, an exception will be raised.

=head2 trained

True if C<train> method have called, false otherwise.

=head1 AUTHOR

Koichi SATOH E<lt>sekia@cpan.orgE<gt>

=head1 SEE ALSO

L<A Short Introduction to Boosting|http://www.site.uottawa.ca/~stan/csi5387/boost-tut-ppr.pdf>

=cut
